import torch
import torch.nn as nn
import torch.nn.functional as F

from nri import MLPEncoder, RNNDecoder, NRITrainingParams, NRI

class LearnableStaticVertexEncoding(nn.Module):
    '''
    learnable positional encoding for vertices in a static graph
    '''

    def __init__(self, nV: int, emb_dim: int):
        '''
        Arguments:
        ---
        - nV: no. of vertex in graph
        - emb_dim: embedding dimension
        '''
        super(LearnableStaticVertexEncoding, self).__init__()
        
        # This embedding layer will act as our learnable positional encoding
        self.positional_embedding = nn.Parameter(torch.empty(nV, emb_dim))
        
        # Initialize the positional embeddings to small values
        nn.init.uniform_(self.positional_embedding, -0.1, 0.1)
    
    def forward(self, b: int) -> torch.Tensor:
        '''
        Arguments:
        ---
        - b: no. of batch

        Returns:
        ---
        - position encoding [B, V, S]
        '''
        return self.positional_embedding.repeat(b, 1, 1)

class MLPEncoderWithPE(MLPEncoder):
    '''
    `MLPEncoder` augmented with learnable position encoding
    '''

    def __init__(
        self, 
        n_in: int, 
        n_hid: int, 
        n_out: int, 
        adj_mat: torch.Tensor,
        do_prob: float = 0.
    ):
        super(MLPEncoderWithPE, self).__init__(
            n_in, 
            n_hid, 
            n_out, 
            adj_mat,
            do_prob
        )

        self.positional_encoding = LearnableStaticVertexEncoding(adj_mat.size(0), n_hid)

    def forward(self, inputs: torch.Tensor):
        # reshape to Tensor[B, V, T * n_in]
        # autoregression
        x = inputs.transpose(1, 2).contiguous().view(inputs.size(0), inputs.size(2), -1)

        # convert node state to node embedding: Tensor[B, V, n_hid]
        x = self.mlp1(x) + self.positional_encoding(x.size(0))

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
    
class RNNDecoderWithPE(RNNDecoder):
    '''
    `RNNDecoder` augmented with positional encoding
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
        super(RNNDecoderWithPE, self).__init__(
            n_in, 
            edge_types, 
            n_hid, 
            adj_mat,
            do_prob, 
            skip_first
        )

        self.positional_encoding = LearnableStaticVertexEncoding(adj_mat.size(0), n_hid)

    def forward(
        self, 
        data: torch.Tensor, 
        rel_type: torch.Tensor, 
        pred_steps: int = 1,
        hidden: torch.Tensor = None,
        train_params: NRITrainingParams = None
    ):
        if hidden is None:
            hidden = self.positional_encoding(data.size(0))

            if data.is_cuda:
                hidden = hidden.cuda()

        return super().forward(data, rel_type, pred_steps, hidden, train_params)

class NRIWithPE(NRI):
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

        self.decoder = RNNDecoderWithPE(
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
    nri = NRIWithPE(state_dim=3, prior_steps=2, adj_mat=adj_mat, edge_types=2, hid_dim=16)
    out = nri.forward(torch.randn((1, 2, 3, 3)), pred_steps=1)