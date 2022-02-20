"""Top-level model classes.

Author:
    Chris Chute (chute@stanford.edu)
"""

import layers
import torch
import torch.nn as nn
import torch.nn.functional as F

class BiDAF(nn.Module):
    """Baseline BiDAF model for SQuAD.

    Based on the paper:
    "Bidirectional Attention Flow for Machine Comprehension"
    by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
    (https://arxiv.org/abs/1611.01603).

    Follows a high-level structure commonly found in SQuAD models:
        - Embedding layer: Embed word indices to get word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, word_vectors, hidden_size, drop_prob=0.):
        super(BiDAF, self).__init__()
        self.emb = layers.Embedding(word_vectors=word_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob)

        self.enc = layers.RNNEncoder(input_size=hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)

        self.att = layers.BiDAFAttention(hidden_size=2 * hidden_size,
                                         drop_prob=drop_prob)

        self.mod = layers.RNNEncoder(input_size=8 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)

        self.out = layers.BiDAFOutput(hidden_size=hidden_size,
                                      drop_prob=drop_prob)

    def forward(self, cw_idxs, qw_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        c_emb = self.emb(cw_idxs)         # (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs)         # (batch_size, q_len, hidden_size)

        c_enc = self.enc(c_emb, c_len)    # (batch_size, c_len, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_len)    # (batch_size, q_len, 2 * hidden_size)

        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)

        mod = self.mod(att, c_len)        # (batch_size, c_len, 2 * hidden_size)

        out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out

class CoAttention(nn.Module):
    def __init__(self, word_vectors, hidden_size, drop_prob=0.2):
        # TODO: write input and output parameters and pass them to layers
        super().__init__()

        # Initialize hyperparameters here
        self.hidden_size = hidden_size
        self.drop_prob = drop_prob

        # Initialize layers here
        self.emb = layers.Embedding(word_vectors=word_vectors, hidden_size=self.hidden_size, drop_prob=self.drop_prob)
        self.lstm_encoder = layers.LSTMEncoder(self.hidden_size, self.hidden_size)
        self.question_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(self.drop_prob)
        self.sentinel_c = nn.Parameter(torch.rand(self.hidden_size,))
        self.sentinel_q = nn.Parameter(torch.rand(self.hidden_size,))

    
    '''
    Input:
        - cw_idxs: Indices of the words in the context.
            Shape (batch_size, context_len,).
        - qw_idxs: Indices of the words in the question.
            Shape (batch_size, question_len,).
    '''
    def forward(self, cw_idxs, qw_idxs):
        batch_size = cw_idxs.size(0)

        # Detect the actual length of the sentences
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        c_emb = self.emb(cw_idxs)         # (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs)         # (batch_size, q_len, hidden_size)

        D = self.dropout(self.lstm_encoder(c_emb, c_len)) # (B, m, l)
        Q = self.dropout(self.lstm_encoder(q_emb, q_len)) # (B, n, l)
        Q = torch.tanh(self.question_proj(Q))
        
        # (l) -> (1, l) -> (B, l) -> (B, 1, l)
        sentinel_c = self.sentinel_c.unsqueeze(0).expand(batch_size, -1).unsqueeze(1)
        sentinel_q = self.sentinel_q.unsqueeze(0).expand(batch_size, -1).unsqueeze(1)

        D = torch.cat((D, sentinel_c), 1) # (B, m+1, l)
        Q = torch.cat((Q, sentinel_q), 1) # (B, n+1, l)
        print("D shape: ", D.shape)
        print("Q shape: ", Q.shape)
        return "Hi"
