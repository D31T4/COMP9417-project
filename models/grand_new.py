'''
NRI with GRAND decoder
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint

from models.nri_PE import MLPEncoderWithPE, RNNDecoderWithPE, NRI

class SingleHeadDiffusivity(nn.Module):
    '''
    diffusivity
    '''
    def __init__(self, n_hid: int, do_prob: float = 0.):
        '''
        Arguments:
        ---
        - n_hid: hidden state dimension
        - do_prob: dropout prob
        '''
        super().__init__()

        self.Wq = nn.Sequential(
            nn.Linear(n_hid, n_hid),
            nn.Dropout(do_prob, inplace=True),
            #nn.Tanh()
        )
        
        self.Wk = nn.Sequential(
            nn.Linear(n_hid, n_hid),
            nn.Dropout(do_prob, inplace=True),
            #nn.Tanh()
        )

        self.scale = n_hid ** -0.5

    def forward(self, x: torch.Tensor):
        '''
        Arguments:
        ---
        - x: tensor[:, V, S]
        
        Returns:
        ---
        - A: tensor[:, V, V]
        '''
        q = self.Wq(x)
        k = self.Wk(x).transpose(1, 2)

        return torch.matmul(q, k) * self.scale

class GraNDDecoder(RNNDecoderWithPE):
    '''
    NRI decoder with GraND
    '''
    def __init__(
        self, 
        n_in: int, 
        edge_types: int, 
        n_hid: int, 
        adj_mat: torch.Tensor,
        do_prob: float = 0., 
        skip_first: bool = False,
        reg: bool = False
    ):
        '''
        Arguments:
        ---
        - reg: apply KE regularization if `True`
        '''
        super(GraNDDecoder, self).__init__(
            n_in=n_in,
            edge_types=edge_types,
            n_hid=n_hid,
            adj_mat=adj_mat,
            do_prob=do_prob,
            skip_first=skip_first
        )

        self.diffusivity_net = nn.ModuleList(SingleHeadDiffusivity(self.n_hid, self.dropout_prob) for _ in range(self.etypes))

        self.t_range = nn.Parameter(torch.tensor([0., 1.]), requires_grad=False)
        
        self.ode_x0_net = nn.Sequential(
            nn.Linear(n_hid, n_hid),
            #nn.Tanh(),
            nn.Dropout(do_prob)
        )
        
        self.ode_const_net = nn.Sequential(
            nn.Linear(n_hid, n_hid),
            #nn.Tanh(),
            nn.Dropout(do_prob)
        )
        
        self.embed_input = nn.Sequential(
            nn.Linear(n_in, n_hid),
            nn.ReLU(inplace=True),
            nn.Dropout(do_prob)
        )

        self.correction_net = nn.Sequential(
            nn.Linear(2 * n_hid, n_hid),
            nn.ReLU(inplace=True),
            nn.Dropout(do_prob)
        )
        

        self.ode_alpha = nn.Parameter(torch.tensor(0.0))
        self.ode_beta = nn.Parameter(torch.tensor(0.0))

        self.reg = reg

    def ode_regularization_term(self, x: torch.Tensor, dx: torch.Tensor):
        '''
        kinetic energy regularization
        '''
        ke = 0.5 * dx.view(dx.shape[0], -1).pow(2).mean(dim=-1)
        return ke

    def ode_with_reg(self, t: torch.Tensor, state: list[torch.Tensor]):
        '''
        diffusion equation with kinetic energy regularization

        stolen from https://github.com/twitter-research/graph-neural-pde/blob/main/src/function_laplacian_diffusion.py

        Arguments:
        ---
        - t: required by ODE solver
        - x: node embedding
        '''
        with torch.enable_grad():
            x = state[0]
            x.requires_grad_(True)
            t.requires_grad_(True)

            ax = torch.matmul(self.ode_weight, x)
            dx = ax + self.ode_const

            reg = self.ode_regularization_term(x, dx)

            return (dx, reg)
        
    def ode(self, t, x):
        '''
        diffusion equation with drift term
        '''
        ax = torch.matmul(self.ode_weight, x)
        dx = self.ode_alpha * ax + self.ode_beta * self.ode_const
        return dx

    def ode_vanilla(self, t, x: torch.Tensor):
        '''
        diffusion equation from GRAND

        stolen from https://github.com/twitter-research/graph-neural-pde/blob/main/src/function_laplacian_diffusion.py
        '''
        ax = torch.matmul(self.ode_weight, x)
        dx = self.ode_alpha * ax + self.ode_beta * self.ode_x0
        return dx
    
    def compute_drift(self, x: torch.Tensor, rel_type: torch.Tensor):
        #return self.ode_const_net(x)
    
        start_idx = int(not self.skip_first_edge_type)

        # node2edge: use hidden state as node embedding
        receivers = x[:, self.recv_edges, :]
        senders = x[:, self.send_edges, :]
        pre_msg = torch.cat([senders, receivers], dim=-1)

        # edge embedding: Tensor[B, E, S]
        all_msgs = torch.zeros((pre_msg.size(0), pre_msg.size(1), self.n_hid), device=x.device)

        # Run separate MLP for every edge type
        for i in range(start_idx, len(self.msg_fc2)):
            msg = F.tanh(self.msg_fc1[i](pre_msg))
            msg = F.dropout(msg, p=self.dropout_prob)
            msg = self.msg_fc2[i](msg)

            all_msgs += msg * rel_type[:, :, i:(i + 1)]

        return self.ode_const_net(x + torch.matmul(self.edge_weights, all_msgs))

    def compute_diffusivity(self, x: torch.Tensor):
        start_idx = int(not self.skip_first_edge_type)

        diffusivity = torch.zeros((x.size(0), x.size(1), x.size(1)), device=x.device)

        # Run separate MLP for every edge type
        for i in range(start_idx, len(self.msg_fc2)):
            diffusivity += self.diffusivity_net[i](x) * self.adj_tensor[:, i, :]

        #diffusivity = F.softmax(diffusivity, dim=-1)

        for i in range(x.size(0)):
            diffusivity[i, :] -= torch.diag(diffusivity[i, :].sum(dim=-1))

        return diffusivity

    def single_step_forward(
        self, 
        inputs: torch.Tensor, 
        rel_type: torch.Tensor, 
        hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        corrected_state = torch.concat([self.embed_input(inputs), hidden], dim=-1)
        corrected_state = self.correction_net(corrected_state)

        self.ode_weight = self.compute_diffusivity(corrected_state)
        self.ode_const = self.compute_drift(corrected_state, rel_type)

        ode_x0 = self.ode_x0_net(corrected_state)
        #self.ode_x0 = ode_x0.clone().detach_()

        if self.reg and self.training:
            reg_states = torch.zeros(hidden.size(0), device=inputs.device)

            state_dt = odeint(
                self.ode_with_reg,
                (ode_x0, reg_states),
                self.t_range,
                method='dopri5',
                rtol=1e-3,
                atol=1e-6
            )

            hidden = state_dt[0][1]
            self.reg_states += state_dt[1][1]
        else:
            hidden = odeint(
                self.ode,
                ode_x0,
                self.t_range,
                method='dopri5',
                rtol=1e-3,
                atol=1e-6
            )[1]

        hidden = F.relu(hidden)

        self.ode_weight = None
        self.ode_const = None

        # GRU-style gated aggregation
        r = F.sigmoid(self.input_r(inputs) + self.hidden_r(hidden))
        i = F.sigmoid(self.input_i(inputs) + self.hidden_i(hidden))
        n = F.tanh(self.input_n(inputs) + r * self.hidden_h(hidden))
        hidden = (1 - i) * n + i * hidden

        # Output MLP
        pred = F.dropout(F.relu(self.out_fc1(hidden)), p=self.dropout_prob)
        pred = F.dropout(F.relu(self.out_fc2(pred)), p=self.dropout_prob)
        pred = self.out_fc3(pred)

        # Predict position/velocity difference
        pred = inputs + pred

        return pred, hidden
    
    def forward(
        self, 
        data: torch.Tensor, 
        rel_type: torch.Tensor, 
        pred_steps: int = 1,
        hidden: torch.Tensor = None,
        train_params = None
    ):
        # adjacent tensor: [B, etype, V, V]
        adj_tensor = torch.zeros((rel_type.size(0), rel_type.size(-1), data.size(2), data.size(2)))

        if self.reg:
            self.reg_states = torch.zeros(data.size(0), device=data.device)
        else:
            self.reg_states = None
        

        for b in range(rel_type.size(0)):
            for e in range(rel_type.size(1)):
                etype = torch.argmax(rel_type[b, e, :])
                adj_tensor[b, etype, self.recv_edges[e], self.send_edges[e]] = 1.

        self.adj_tensor = adj_tensor

        if data.is_cuda:
            self.adj_tensor = self.adj_tensor.cuda()

        out = super().forward(data, rel_type, pred_steps, hidden, train_params)

        if self.reg:
            self.reg_states /= pred_steps

        return out

class GraNRI(NRI):
    '''
    NRI with GRAND decoder
    '''
    def __init__(
        self, 
        state_dim: int, 
        prior_steps: int, 
        adj_mat: torch.Tensor, 
        do_prob: float = 0.,
        edge_types: int = 4,
        enc_hid_dim: int = 256,
        dec_hid_dim: int = 256,
        gumbel_temp: float = 0.5,
        reg: bool = False
    ):
        super(NRI, self).__init__()

        self.encoder = MLPEncoderWithPE(
            n_in=prior_steps * state_dim,
            n_hid=enc_hid_dim,
            n_out=edge_types,
            adj_mat=adj_mat,
            do_prob=do_prob
        )

        self.decoder = GraNDDecoder(
            n_in=state_dim,
            edge_types=edge_types,
            n_hid=dec_hid_dim,
            adj_mat=adj_mat,
            do_prob=do_prob,
            reg=reg
        )

        self.gumbel_temp = gumbel_temp
        self.prior_steps = prior_steps

if __name__ == '__main__':
    # run test
    import sys
    sys.path.insert(0, '..')

    adj_mat = torch.ones((3, 3)) - torch.eye(3)    
    nri = GraNRI(state_dim=3, prior_steps=2, adj_mat=adj_mat, edge_types=2, hid_dim=16)
    out = nri.forward(torch.randn((1, 2, 3, 3)), pred_steps=1)