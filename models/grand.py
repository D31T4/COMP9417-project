import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint

from .nri_PE import MLPEncoderWithPE, RNNDecoderWithPE, NRI

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
        skip_first: bool = False
    ):
        super(GraNDDecoder, self).__init__(
            n_in=n_in,
            edge_types=edge_types,
            n_hid=n_hid,
            adj_mat=adj_mat,
            do_prob=do_prob,
            skip_first=skip_first
        )

        # ode params
        self.ode_alpha = nn.Parameter(torch.tensor(0.0))
        self.ode_beta = nn.Parameter(torch.tensor(0.0))
        self.ode_x0: torch.Tensor = 0

        # attention net
        self.Wq = nn.ModuleList(nn.Linear(self.n_hid, self.n_hid) for _ in range(self.etypes))
        self.Wk = nn.ModuleList(nn.Linear(self.n_hid, self.n_hid) for _ in range(self.etypes))


    def ode(self, t, x: torch.Tensor):
        '''
        diffusion equation.

        stolen from https://github.com/twitter-research/graph-neural-pde/blob/main/src/function_laplacian_diffusion.py

        Arguments:
        ---
        - t: required by ODE solver
        - x: node embedding
        '''
        ax = torch.matmul(self.ode_attention, x)
        x = self.ode_alpha * (ax - x)
        return x + self.ode_beta * self.ode_x0

    def single_step_forward(
        self, 
        inputs: torch.Tensor, 
        rel_type: torch.Tensor, 
        hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:

        start_idx = int(not self.skip_first_edge_type)

        diffusivity = torch.zeros((hidden.size(0), hidden.size(1), hidden.size(1)))

        if inputs.is_cuda:
            diffusivity = diffusivity.cuda()

        # Run separate MLP for every edge type
        for i in range(start_idx, len(self.msg_fc2)):
            q = F.relu(self.Wq[i](hidden))
            k = F.relu(self.Wk[i](hidden)).transpose(1, 2)

            diffusivity += torch.matmul(q, k) * self.adj_tensor[:, i, :]

        #attention = F.softmax(attention, dim=-1)


        self.ode_x0 = hidden.clone().detach_()
        self.ode_attention = diffusivity

        t = torch.tensor([0., 1.])

        if inputs.is_cuda:
            t = t.cuda()

        node_embeddings = odeint(
            self.ode,
            hidden,
            t,
            method='dopri5',
            rtol=1e-3,
            atol=1e-6
        )[1]

        node_embeddings = F.relu(node_embeddings)

        # GRU-style gated aggregation
        r = F.sigmoid(self.input_r(inputs) + self.hidden_r(node_embeddings))
        i = F.sigmoid(self.input_i(inputs) + self.hidden_i(node_embeddings))
        n = F.tanh(self.input_n(inputs) + r * self.hidden_h(node_embeddings))
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

        for b in range(rel_type.size(0)):
            for e in range(rel_type.size(1)):
                etype = torch.argmax(rel_type[b, e, :])
                adj_tensor[b, etype, self.recv_edges[e], self.send_edges[e]] = 1.

        # scale by 1 / sqrt(d) to compute scaled dot-product attention
        self.adj_tensor = adj_tensor * self.n_hid ** -0.5

        if data.is_cuda:
            self.adj_tensor = self.adj_tensor.cuda()

        return super().forward(data, rel_type, pred_steps, hidden, train_params)

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
        hid_dim: int = 256,
        gumbel_temp: float = 0.5,
    ):
        super(NRI, self).__init__()

        self.encoder = MLPEncoderWithPE(
            n_in=prior_steps * state_dim,
            n_hid=hid_dim,
            n_out=edge_types,
            adj_mat=adj_mat,
            do_prob=do_prob
        )

        self.decoder = GraNDDecoder(
            n_in=state_dim,
            edge_types=edge_types,
            n_hid=hid_dim,
            adj_mat=adj_mat,
            do_prob=do_prob
        )

        self.gumbel_temp = gumbel_temp
        self.prior_steps = prior_steps

if __name__ == '__main__':
    # run test
    adj_mat = torch.ones((3, 3)) - torch.eye(3)    
    nri = GraNRI(state_dim=3, prior_steps=2, adj_mat=adj_mat, edge_types=2, hid_dim=16)
    out = nri.forward(torch.randn((1, 2, 3, 3)), pred_steps=1)