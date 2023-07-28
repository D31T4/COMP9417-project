import tensorflow as tf
import torch
def create_look_ahead_mask(size_q, size_k):
  mask = 1 - tf.linalg.band_part(tf.ones((size_q, size_k)), -1, 0)
  return mask  # (seq_len, seq_len)

def scaled_dot_product_attention(q, k, v, mask):
  """Calculate the attention weights.
  q, k, v must have matching leading dimensions.
  k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
  The mask has different shapes depending on its type(padding or look ahead) 
  but it must be broadcastable for addition.
  
  Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Any value to create and use a lookahead mask. 
          Defaults to None.
    
  Returns:
    output, attention_weights
  """

  matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
  
  # scale matmul_qk
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

  # add the mask to the scaled tensor.
  if mask is not None:
    mask = create_look_ahead_mask(tf.shape(q)[-2], tf.shape(k)[-2])
    scaled_attention_logits += (mask * -1e9)  

  # softmax is normalized on the last axis (seq_len_k) so that the scores
  # add up to 1.
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

  output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

  return output, attention_weights

class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model
    
    assert d_model % self.num_heads == 0
    
    self.depth = d_model // self.num_heads
    
    self.wq = tf.keras.layers.Dense(d_model)
    self.wk = tf.keras.layers.Dense(d_model)
    self.wv = tf.keras.layers.Dense(d_model)
    
    self.dense = tf.keras.layers.Dense(d_model)
        
  def split_heads(self, x, batch_size):
    """Split the last dimension into (num_heads, depth).
    Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
    """
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])
    
  def call(self, v, k, q, mask):
    batch_size = tf.shape(q)[0]
    
    q = self.wq(q)  # (batch_size, seq_len, d_model)
    k = self.wk(k)  # (batch_size, seq_len, d_model)
    v = self.wv(v)  # (batch_size, seq_len, d_model)
    
    q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
    k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
    v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
    
    # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    scaled_attention, attention_weights = scaled_dot_product_attention(
        q, k, v, mask)
    
    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

    concat_attention = tf.reshape(scaled_attention, 
                                  (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

    output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
        
    return output, attention_weights
  

def point_wise_feed_forward_network(d_model, dff):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      # tf.keras.activations.gelu(approximate=False), # tf-nightly
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
  ])

class DecoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(DecoderLayer, self).__init__()

    self.mha = MultiHeadAttention(d_model, num_heads)
    self.ffn = point_wise_feed_forward_network(d_model, dff)

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    
    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)
    
  def call(self, x, rel, training, mask):
    # t.shape == (batch_size, input_seq_len, d_model) 
    # for t in [x, out1, attn_out, res1, out2, ffn_out, res2]
    
    # masked multihead self-attention with residual connection
    out1 = self.layernorm1(self.dropout1(x, training=training)) 
    attn_out, attn_weights = self.mha(out1, out1, out1, mask) 
    res1 = x + attn_out

    out1 = self.layernorm1(self.dropout1(res1, training=training)) 
    attn_out, attn_weights = self.mha(rel, rel, out1, None) 
    res1 = out1 + attn_out

    # feed forward neural network with residual connection
    out2 = self.layernorm2(self.dropout2(res1, training=training))
    ffn_out = self.ffn(out2)
    res2 = res1 + ffn_out
    
    return res2, attn_weights
  
class DecoderBlock(tf.keras.layers.Layer):
  def __init__(self, num_layers, d_model, num_heads, dff, vocab_size,
               maximum_position_encoding, rate=0.1):
    super(DecoderBlock, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers
    
    self.tok_embedding = tf.keras.layers.Embedding(vocab_size, d_model)
    self.pos_embedding = tf.keras.layers.Embedding(maximum_position_encoding, d_model)                                            
        
    self.enc_layers = [DecoderLayer(d_model, num_heads, dff, rate) 
                       for _ in range(num_layers)]
  
    self.dropout = tf.keras.layers.Dropout(rate)
    self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
  def call(self, x, rel, training, mask):

    batch = tf.shape(x)[0]
    seq_len = tf.shape(x)[1]
    attention_weights = {}  

    # token and positional embedding
    positions = tf.expand_dims(tf.range(seq_len), axis=0)
    #print(x[0])
    #x = self.tok_embedding(x)  # (batch_size, input_seq_len, d_model)
    #print(x[0][0])
    x = x + tf.tile(self.pos_embedding(positions), [batch, 1, 1])  # (batch_size, input_seq_len, d_model)
    
    # n * decoder layers

    #pred = []

    for i in range(self.num_layers):
      x, attn_w = self.enc_layers[i](x, rel, training, mask)
      attention_weights['decoder_layer{}'.format(i+1)] = attn_w
      #pred.append(x)

    # dropout and layer normalization
    x = self.dropout(x, training=training)
    x = self.layer_norm(x)

    #pred = self.dropout(pred, training=training)
    #pred = self.layer_norm(pred)

    #preds = torch.stack(pred, dim=1)
              
    return x, attention_weights  # (batch_size, input_seq_len, d_model)


class ImageTransformer(tf.keras.Model):
  def __init__(self, num_layers, d_model, num_heads, dff,
               vocab_size, max_pos_encoding, rate=0.1):
    super(ImageTransformer, self).__init__()

    self.decoder = DecoderBlock(num_layers, d_model, num_heads, dff, 
                           vocab_size, max_pos_encoding, rate)

    self.final_layer = tf.keras.layers.Dense(vocab_size)
    
  def call(self, inp, tar, training, mask):

    enc_output, attention_weights = self.decoder(inp, tar, training, mask)  # (batch_size, inp_seq_len, d_model)    
    
    final_output = self.final_layer(enc_output)  # (batch_size, tar_seq_len, target_vocab_size)
    
    return final_output, attention_weights
  

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

        #print(f'x first: {x}')
        x = self.mlp1(x)

        #print(f'x second: {x}')

        # compute edge embedding
        x = self.node2edge(x) # aggregate connected nodes: Tensor[B, E, 2 * n_hid]
        #print(f'x transform {x}')
        x = self.mlp2(x) # processed edge embedding: Tensor[B, E, n_hid]
        x_skip = x # residual

        #print(f'x third: {x}')

        # node embedding
        x = self.edge2node(x)
        x = self.mlp3(x)

        #print(f'x forth: {x}')

        # edge embedding
        x = self.node2edge(x)
        x = torch.cat((x, x_skip), dim=-1) # Tensor[B, E, 3 * n_hid]
        x = self.mlp4(x)

        #print(self.fc_out(x))

        tmp = self.edge2node(x)
        #print(f'tmp ==: {tmp}')

        # edge latent
        return self.fc_out(tmp)
    
class NRITrainingParams:
    def __init__(self, ground_truth_interval: int):
        self.ground_truth_interval = ground_truth_interval



class NRItf(nn.Module):
    def __init__(
        self, 
        state_dim: int, 
        prior_steps: int, 
        adj_mat: torch.Tensor, 
        do_prob: float = 0.,
        edge_types: int = 16,
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
        super(NRItf, self).__init__()

        self.encoder = MLPEncoder(
            n_in=prior_steps * state_dim,
            n_hid=hid_dim,
            n_out=edge_types,
            adj_mat=adj_mat,
            do_prob=do_prob
        )

        '''self.decoder = RNNDecoder(
            n_in=state_dim,
            edge_types=edge_types,
            n_hid=hid_dim,
            adj_mat=adj_mat,
            do_prob=do_prob
        )'''

        self.gumbel_temp = gumbel_temp
        self.prior_steps = prior_steps
        self.dm = edge_types

        self.tf_decoder = ImageTransformer(
            num_layers=2, d_model=edge_types, num_heads=2, dff=64,
            vocab_size=3, max_pos_encoding=784
        )

    def forward(self, data: torch.Tensor, pred_steps: int, rand: bool = False, train_params: NRITrainingParams = None):
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
        #print(data[:, :self.prior_steps, :])

        logits = self.encoder(data[:, :self.prior_steps, :])

        #print(logits)

        '''if rand or self.training:
            rel_types = F.gumbel_softmax(logits, tau=self.gumbel_temp, hard=True)
        else:
            rel_types = F.softmax(logits, dim=-1)
            index = rel_types.max(-1, keepdim=True)
            rel_types = torch.zeros_like(logits).scatter_(-1, index, 1.0)'''

        #print(data)
        #print(logits)

        x = data.transpose(1, 2).contiguous().view(data.size(0), data.size(2), -1)
        sxf = nn.Linear(x.shape[2], self.dm)
        x = sxf(x)
        #print(logits)
        
        out, _ = self.tf_decoder(x.detach().numpy(), logits.detach().numpy(), training = True, mask = True)
        #print(type(out))
        return out, logits



