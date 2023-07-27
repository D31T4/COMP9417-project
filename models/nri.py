import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_graph_stats(adj_mat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    '''
    compute graph statistics

    Arguments:
    ---
    - adjacency matrix
    
    Returns:
    ---
    - send edges
    - recv edges
    - edge weight
    '''
    V = adj_mat.shape[0]
    deg = torch.sum(adj_mat, dim=1)

    adj_mat = adj_mat.numpy()

    send_edges, recv_edges = np.where(adj_mat)
    E = send_edges.shape[0]

    # vertex => edge one-hot vector. True when edge point to node
    edge_adj: torch.Tensor = torch.zeros((V, E))

    send_edges = torch.tensor(send_edges)
    recv_edges = torch.tensor(recv_edges)

    for v in range(V):
        edge_adj[v, :] = torch.eq(recv_edges, v)

    # normalize onehot and we get adjcency matrix
    # similar to normalization in GCN (D^-0.5 @ A @ D^-0.5), NRI does D^-1 @ A
    edge_adj = edge_adj / deg[:, None]

    return send_edges, recv_edges, edge_adj

class MLP(nn.Module):
    def __init__(
        self, 
        n_in: int, 
        n_hid: int, 
        n_out: int, 
        do_prob: float = 0.
    ):
        '''
        Arguments:
        ---
        - n_in: input size
        - n_hid: hidden layer size
        - n_out: output size
        - do_prob: dropout probability
        '''
        super(MLP, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(n_in, n_hid),
            nn.ELU(inplace=True),
            nn.Dropout(do_prob),
            nn.Linear(n_hid, n_out),
            nn.ELU(inplace=True)
        )

        self.bn = nn.BatchNorm1d(n_out)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs: torch.Tensor):
        '''
        Arguments:
        ---
        - inputs: Tensor[B, N, S], where B = no. of batch, N = no. of elements. S = dim of state

        Returns:
        ---
        - Tensor[B, N, S]
        '''
        # element-wise batch norm
        x: torch.Tensor = inputs.view(-1, inputs.size(-1))
        x = self.bn(x)
        return x.view(inputs.shape)
    
    def forward(self, inputs: torch.Tensor):
        '''
        Arguments:
        ---
        - inputs: Tensor[B, N, n_in]

        Returns:
        ---
        - output: Tensor[B, N, n_out]
        '''
        x = self.model(inputs)
        return self.batch_norm(x)
    
class MLPEncoder(nn.Module):
    '''
    An autoregressive encoder with fixed window length
    '''

    def __init__(
        self, 
        n_in: int, 
        n_hid: int, 
        n_out: int, 
        adj_mat: torch.Tensor,
        do_prob: float = 0.
    ):
        '''
        Arguments:
        ---
        - n_in: input size (prior size = T * S) where T = no. of timesteps and S = dim of node state
        - n_hid: hidden layer size
        - n_out: output size
        - adj_mat: adjacency matrix
        - do_prob: dropout probability
        '''
        super(MLPEncoder, self).__init__()

        self.mlp1 = MLP(n_in, n_hid, n_hid, do_prob)
        self.mlp2 = MLP(n_hid * 2, n_hid, n_hid, do_prob)
        self.mlp3 = MLP(n_hid, n_hid, n_hid, do_prob)
        self.mlp4 = MLP(n_hid * 3, n_hid, n_hid, do_prob)

        self.fc_out = nn.Linear(n_hid, n_out)

        self.init_weights()

        self.send_edges, self.recv_edges, self.edge_weights = compute_graph_stats(adj_mat)
        self.send_edges = nn.Parameter(self.send_edges, requires_grad=False)
        self.recv_edges = nn.Parameter(self.recv_edges, requires_grad=False)
        self.edge_weights = nn.Parameter(self.edge_weights, requires_grad=False)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def edge2node(self, edge_embeddings: torch.Tensor) -> torch.Tensor:
        '''
        compute node embeddings by averaging embeddings of connected edges.
        basically a graph convolution with a ones filter
        
        Arguments:
        ---
        - edge_embeddings: edge embeddings [B, E, n_hid]

        Returns:
        ---
        - node_embeddings: node embeddings [B, V, n_hid]
        '''
        # node aggregate from in-edges like GCN
        return torch.matmul(self.edge_weights, edge_embeddings)
    
    def node2edge(self, node_embeddings: torch.Tensor) -> torch.Tensor:
        '''
        compute edge embedding by concat sending node and receiving node embeddings

        Arguments:
        ---
        - node_embeddings: node embedding [B, V, n_hid]

        Returns:
        ---
        - edge embedding [B, E, 2 * n_hid]
        '''
        send_embed = node_embeddings[:, self.send_edges, :]
        recv_embed = node_embeddings[:, self.recv_edges, :]
        edges = torch.cat([send_embed, recv_embed], dim=2)
        return edges
    
    def forward(self, inputs: torch.Tensor):
        '''
        Arguments:
        ---
        - inputs: Tensor[B, T, V, n_in]; where B = batch size, T = no. of timesteps, V = no. of vertex

        Returns:
        ---
        - edge latent: Tensor[B, E, n_out]
        '''
        # reshape to Tensor[B, V, T * n_in]
        # autoregression
        x = inputs.transpose(1, 2).contiguous().view(inputs.size(0), inputs.size(2), -1)

        # convert node state to node embedding: Tensor[B, V, n_hid]
        x = self.mlp1(x)

        # compute edge embedding
        x = self.node2edge(x) # aggregate connected nodes: Tensor[B, E, 2 * n_hid]
        x = self.mlp2(x) # processed edge embedding: Tensor[B, E, n_hid]
        x_skip = x # residual

        # node embedding
        x = self.edge2node(x)
        x = self.mlp3(x)

        # edge embedding
        x = self.node2edge(x)
        x = torch.cat((x, x_skip), dim=-1) # Tensor[B, E, 3 * n_hid]
        x = self.mlp4(x)

        # edge latent
        return self.fc_out(x)
    
class NRITrainingParams:
    def __init__(self, ground_truth_interval: int):
        self.ground_truth_interval = ground_truth_interval

class RNNDecoder(nn.Module):
    def __init__(
        self, 
        n_in: int, 
        edge_types: int, 
        n_hid: int, 
        adj_mat: torch.Tensor,
        do_prob: float = 0., 
        skip_first: bool = False
    ):
        '''
        Arguments:
        ---
        - n_in: node state size (input, output)
        - edge_types: no. of edge types
        - n_hid: hidden layer size
        - do_prob: dropout probability
        - skip_first
        '''
        super(RNNDecoder, self).__init__()

        self.msg_fc1 = nn.ModuleList([
            nn.Linear(2 * n_hid, n_hid) for _ in range(edge_types)
        ])

        self.msg_fc2 = nn.ModuleList([
            nn.Linear(n_hid, n_hid) for _ in range(edge_types)
        ])

        self.n_hid = n_hid
        self.etypes = edge_types
        self.skip_first_edge_type = skip_first

        # GRU
        self.hidden_r = nn.Linear(n_hid, n_hid, bias=False)
        self.hidden_i = nn.Linear(n_hid, n_hid, bias=False)
        self.hidden_h = nn.Linear(n_hid, n_hid, bias=False)

        # GRU
        self.input_r = nn.Linear(n_in, n_hid, bias=True)
        self.input_i = nn.Linear(n_in, n_hid, bias=True)
        self.input_n = nn.Linear(n_in, n_hid, bias=True)

        self.out_fc1 = nn.Linear(n_hid, n_hid)
        self.out_fc2 = nn.Linear(n_hid, n_hid)
        self.out_fc3 = nn.Linear(n_hid, n_in)

        self.dropout_prob = do_prob

        self.send_edges, self.recv_edges, self.edge_weights = compute_graph_stats(adj_mat)
        self.send_edges = nn.Parameter(self.send_edges, requires_grad=False)
        self.recv_edges = nn.Parameter(self.recv_edges, requires_grad=False)
        self.edge_weights = nn.Parameter(self.edge_weights, requires_grad=False)

    def single_step_forward(
        self, 
        inputs: torch.Tensor, 
        rel_type: torch.Tensor, 
        hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        '''
        Arguments:
        ---
        - inputs: Tensor[B, V, S]
        - rel_type: relation type onehot. Tensor[B, E, etype]
        - hidden: hidden state of GRU. Tensor[B, V, n_hid]

        Returns:
        ---
        - prediction at t + 1
        - hidden state at t + 1
        '''
        # node2edge: use hidden state as node embedding
        receivers = hidden[:, self.recv_edges, :]
        senders = hidden[:, self.send_edges, :]
        pre_msg = torch.cat([senders, receivers], dim=-1)

        # edge embedding: Tensor[B, E, S]
        all_msgs = torch.zeros(pre_msg.size(0), pre_msg.size(1), self.n_hid)

        if inputs.is_cuda:
            all_msgs = all_msgs.cuda()


        start_idx = int(not self.skip_first_edge_type)

        # Run separate MLP for every edge type
        # sum edge embedding for each edge type
        for i in range(start_idx, len(self.msg_fc2)):
            msg = pre_msg * rel_type[:, :, i:(i + 1)]

            msg = F.tanh(self.msg_fc1[i](msg))
            msg = F.dropout(msg, p=self.dropout_prob)
            msg = F.tanh(self.msg_fc2[i](msg))

            all_msgs += msg

        # edge2node aggregate
        agg_msgs = torch.matmul(self.edge_weights, all_msgs)

        # GRU-style gated aggregation
        r = F.sigmoid(self.input_r(inputs) + self.hidden_r(agg_msgs))
        i = F.sigmoid(self.input_i(inputs) + self.hidden_i(agg_msgs))
        n = F.tanh(self.input_n(inputs) + r * self.hidden_h(agg_msgs))
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
        train_params: NRITrainingParams = None
    ):
        '''
        Arguments:
        ---
        - data: tensor[B, T, V, S]
        - rel_type: relation type onehot. tensor[B, E, etype]

        Returns:
        ---
        - predictions: tensor[B, T, V, S]
        '''
        if hidden is None:
            hidden = torch.zeros(data.size(0), data.size(2), self.n_hid)

            if data.is_cuda:
                hidden = hidden.cuda()

        pred_all = []

        for step in range(0, pred_steps):
            if step == 0 or (self.training and train_params and step % train_params.ground_truth_interval == 0):
                ins = data[:, step, :]
            else:
                ins = pred_all[-1]

            pred, hidden = self.single_step_forward(ins, rel_type, hidden)
            pred_all.append(pred)

        preds = torch.stack(pred_all, dim=1)
        return preds, hidden

class NRI(nn.Module):
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
        '''
        Arguments:
        ---
        - state_dim: node state dimension
        - prior_steps: no. of prior steps for encoder
        - adj_mat: adjacency matrix
        - do_prob: dropout prob.
        - edge_types: no. of edge types
        - hid_dim: hidden state dimension
        - gumbel_temp: temperature parameter for sampler
        '''
        super(NRI, self).__init__()

        self.encoder = MLPEncoder(
            n_in=prior_steps * state_dim,
            n_hid=hid_dim,
            n_out=edge_types,
            adj_mat=adj_mat,
            do_prob=do_prob
        )

        self.decoder = RNNDecoder(
            n_in=state_dim,
            edge_types=edge_types,
            n_hid=hid_dim,
            adj_mat=adj_mat,
            do_prob=do_prob
        )

        self.gumbel_temp = gumbel_temp
        self.prior_steps = prior_steps

    def forward(self, data: torch.Tensor, pred_steps: int, rand: bool = False, train_params: NRITrainingParams = None) -> tuple[torch.Tensor, torch.Tensor]:
        '''
        Arguments:
        ---
        - data: tensor[B, T, V, state_dim]
        - pred_steps: no. of steps to be predicted
        - rand: sample edge types randomly (only applicable in eval mode)
        - train_params: training params (only applicable in train mode)

        Returns:
        ---
        - predictions: tensor[B, pred_steps, V, state_dim]
        '''
        logits = self.encoder(data[:, :self.prior_steps, :])

        if rand or self.training:
            rel_types = F.gumbel_softmax(logits, tau=self.gumbel_temp, hard=True)
        else:
            rel_types = F.softmax(logits, dim=-1)
            index = rel_types.max(-1, keepdim=True)[1]
            rel_types = torch.zeros_like(logits).scatter_(-1, index, 1.0)

        out, _ = self.decoder(data, rel_types, pred_steps, train_params=train_params)
        return out, logits


if __name__ == '__main__':
    # run test
    adj_mat = torch.ones((3, 3)) - torch.eye(3)    
    nri = NRI(state_dim=3, prior_steps=2, adj_mat=adj_mat, edge_types=2, hid_dim=16)
    out = nri.forward(torch.randn((1, 2, 3, 3)), pred_steps=1)