"""Assortment of layers for use in models.py.

Author:
    Chris Chute (chute@stanford.edu)
"""

from unicodedata import bidirectional
from numpy import argmax
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from util import masked_softmax


class Embedding(nn.Module):
    """Embedding layer used by BiDAF, without the character-level component.

    Word-level embeddings are further refined using a 2-layer Highway Encoder
    (see `HighwayEncoder` class for details).

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations
    """
    def __init__(self, word_vectors, hidden_size, drop_prob):
        super(Embedding, self).__init__()
        self.drop_prob = drop_prob
        self.embed = nn.Embedding.from_pretrained(word_vectors)
        self.proj = nn.Linear(word_vectors.size(1), hidden_size, bias=False)
        self.hwy = HighwayEncoder(2, hidden_size)

    def forward(self, x):
        emb = self.embed(x)   # (batch_size, seq_len, embed_size)
        emb = F.dropout(emb, self.drop_prob, self.training)
        emb = self.proj(emb)  # (batch_size, seq_len, hidden_size)
        emb = self.hwy(emb)   # (batch_size, seq_len, hidden_size)

        return emb


class HighwayEncoder(nn.Module):
    """Encode an input sequence using a highway network.

    Based on the paper:
    "Highway Networks"
    by Rupesh Kumar Srivastava, Klaus Greff, Jürgen Schmidhuber
    (https://arxiv.org/abs/1505.00387).

    Args:
        num_layers (int): Number of layers in the highway encoder.
        hidden_size (int): Size of hidden activations.
    """
    def __init__(self, num_layers, hidden_size):
        super(HighwayEncoder, self).__init__()
        self.transforms = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                         for _ in range(num_layers)])
        self.gates = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                    for _ in range(num_layers)])

    def forward(self, x):
        for gate, transform in zip(self.gates, self.transforms):
            # Shapes of g, t, and x are all (batch_size, seq_len, hidden_size)
            g = torch.sigmoid(gate(x))
            t = F.relu(transform(x))
            x = g * t + (1 - g) * x

        return x


class RNNEncoder(nn.Module):
    """General-purpose layer for encoding a sequence using a bidirectional RNN.

    Encoded output is the RNN's hidden state at each position, which
    has shape `(batch_size, seq_len, hidden_size * 2)`.

    Args:
        input_size (int): Size of a single timestep in the input.
        hidden_size (int): Size of the RNN hidden state.
        num_layers (int): Number of layers of RNN cells to use.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 drop_prob=0.):
        super(RNNEncoder, self).__init__()
        self.drop_prob = drop_prob
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True,
                           bidirectional=True,
                           dropout=drop_prob if num_layers > 1 else 0.)

    def forward(self, x, lengths):
        # Save original padded length for use by pad_packed_sequence
        orig_len = x.size(1)

        # Sort by length and pack sequence for RNN
        lengths, sort_idx = lengths.sort(0, descending=True)
        x = x[sort_idx]     # (batch_size, seq_len, input_size)
        x = pack_padded_sequence(x, lengths.cpu(), batch_first=True)

        # Apply RNN
        x, _ = self.rnn(x)  # (batch_size, seq_len, 2 * hidden_size)

        # Unpack and reverse sort
        x, _ = pad_packed_sequence(x, batch_first=True, total_length=orig_len)
        _, unsort_idx = sort_idx.sort(0)
        x = x[unsort_idx]   # (batch_size, seq_len, 2 * hidden_size)

        # Apply dropout (RNN applies dropout after all but the last layer)
        x = F.dropout(x, self.drop_prob, self.training)

        return x


class BiDAFAttention(nn.Module):
    """Bidirectional attention originally used by BiDAF.

    Bidirectional attention computes attention in two directions:
    The context attends to the query and the query attends to the context.
    The output of this layer is the concatenation of [context, c2q_attention,
    context * c2q_attention, context * q2c_attention]. This concatenation allows
    the attention vector at each timestep, along with the embeddings from
    previous layers, to flow through the attention layer to the modeling layer.
    The output has shape (batch_size, context_len, 8 * hidden_size).

    Args:
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob=0.1):
        super(BiDAFAttention, self).__init__()
        self.drop_prob = drop_prob
        self.c_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.q_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.cq_weight = nn.Parameter(torch.zeros(1, 1, hidden_size))
        for weight in (self.c_weight, self.q_weight, self.cq_weight):
            nn.init.xavier_uniform_(weight)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, c, q, c_mask, q_mask):
        batch_size, c_len, _ = c.size()
        q_len = q.size(1)
        s = self.get_similarity_matrix(c, q)        # (batch_size, c_len, q_len)
        c_mask = c_mask.view(batch_size, c_len, 1)  # (batch_size, c_len, 1)
        q_mask = q_mask.view(batch_size, 1, q_len)  # (batch_size, 1, q_len)
        s1 = masked_softmax(s, q_mask, dim=2)       # (batch_size, c_len, q_len)
        s2 = masked_softmax(s, c_mask, dim=1)       # (batch_size, c_len, q_len)

        # (bs, c_len, q_len) x (bs, q_len, hid_size) => (bs, c_len, hid_size)
        a = torch.bmm(s1, q)
        # (bs, c_len, c_len) x (bs, c_len, hid_size) => (bs, c_len, hid_size)
        b = torch.bmm(torch.bmm(s1, s2.transpose(1, 2)), c)

        x = torch.cat([c, a, c * a, c * b], dim=2)  # (bs, c_len, 4 * hid_size)

        return x

    def get_similarity_matrix(self, c, q):
        """Get the "similarity matrix" between context and query (using the
        terminology of the BiDAF paper).

        A naive implementation as described in BiDAF would concatenate the
        three vectors then project the result with a single weight matrix. This
        method is a more memory-efficient implementation of the same operation.

        See Also:
            Equation 1 in https://arxiv.org/abs/1611.01603
        """
        c_len, q_len = c.size(1), q.size(1)
        c = F.dropout(c, self.drop_prob, self.training)  # (bs, c_len, hid_size)
        q = F.dropout(q, self.drop_prob, self.training)  # (bs, q_len, hid_size)

        # Shapes: (batch_size, c_len, q_len)
        s0 = torch.matmul(c, self.c_weight).expand([-1, -1, q_len])
        s1 = torch.matmul(q, self.q_weight).transpose(1, 2)\
                                           .expand([-1, c_len, -1])
        s2 = torch.matmul(c * self.cq_weight, q.transpose(1, 2))
        s = s0 + s1 + s2 + self.bias

        return s


class BiDAFOutput(nn.Module):
    """Output layer used by BiDAF for question answering.

    Computes a linear transformation of the attention and modeling
    outputs, then takes the softmax of the result to get the start pointer.
    A bidirectional LSTM is then applied the modeling output to produce `mod_2`.
    A second linear+softmax of the attention output and `mod_2` is used
    to get the end pointer.

    Args:
        hidden_size (int): Hidden size used in the BiDAF model.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob):
        super(BiDAFOutput, self).__init__()
        self.att_linear_1 = nn.Linear(8 * hidden_size, 1)
        self.mod_linear_1 = nn.Linear(2 * hidden_size, 1)

        self.rnn = RNNEncoder(input_size=2 * hidden_size,
                              hidden_size=hidden_size,
                              num_layers=1,
                              drop_prob=drop_prob)

        self.att_linear_2 = nn.Linear(8 * hidden_size, 1)
        self.mod_linear_2 = nn.Linear(2 * hidden_size, 1)

    def forward(self, att, mod, mask):
        # Shapes: (batch_size, seq_len, 1)
        logits_1 = self.att_linear_1(att) + self.mod_linear_1(mod)
        mod_2 = self.rnn(mod, mask.sum(-1))
        logits_2 = self.att_linear_2(att) + self.mod_linear_2(mod_2)

        # Shapes: (batch_size, seq_len)
        log_p1 = masked_softmax(logits_1.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(logits_2.squeeze(), mask, log_softmax=True)

        return log_p1, log_p2


class LSTMEncoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, batch_first=True)

    def forward(self, x, lengths):
        orig_len = x.size(1)

        lengths, sort_idx = lengths.sort(0, descending=True)
        x = x[sort_idx]
        x = pack_padded_sequence(x, lengths.cpu(), batch_first=True)
        x, _ = self.lstm(x)

        x, _ = pad_packed_sequence(x, batch_first=True, total_length=orig_len)
        _, unsort_idx = sort_idx.sort(0)
        x = x[unsort_idx]

        # TODO: determine whether or not we need dropout here 
        # (I guess no since we can do it outside)

        return x

class FusionBiLSTM(nn.Module):
    def __init__(self, hidden_dim):
        super(FusionBiLSTM, self).__init__()
        self.fusion_bilstm = nn.LSTM(3 * hidden_dim, hidden_dim, 1, batch_first=True, bidirectional=True)

    def forward(self, x, lengths):
        orig_len = x.size(1)

        lengths, sort_idx = lengths.sort(0, descending=True)
        x = x[sort_idx]
        x = pack_padded_sequence(x, lengths.cpu(), batch_first=True)
        x, _ = self.fusion_bilstm(x)

        x, _ = pad_packed_sequence(x, batch_first=True, total_length=orig_len)
        _, unsort_idx = sort_idx.sort(0)
        x = x[unsort_idx]
        return x

class DynamicDecoder(nn.Module):
    def __init__(self, hidden_dim, pooling_size, max_iter_num):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.decoder = nn.LSTMCell(hidden_dim * 4, hidden_dim, bidirectional=False)
        self.max_iter_num = max_iter_num
        self.start = HighMaxoutNetwork(self.hidden_dim, pooling_size)
        self.end = HighMaxoutNetwork(self.hidden_dim, pooling_size)

    def forward(self, U, true_answer):
        b, m, l = U.size()
        true_start = true_answer[:, 0]
        true_end = true_answer[:, 1]
        prev_start = 0
        prev_end = m - 1
        h = torch.zeros((b, l))
        for _ in range(self.max_iter_num):
            prev_start = self.start.forward(U, h, prev_start, prev_end).argmax(-1)
            prev_end = self.end.forward(U, h, prev_start, prev_end).argmax(-1)
            loss = self.start.loss(prev_start, true_start) + self.end.loss(prev_end, true_end)

        return loss, prev_start, prev_end

class HighMaxoutNetwork(nn.Module):
    def __init__(self, hidden_dim, pooling_size):
        super().__init__()
        self.embedding_dim = hidden_dim
        self.r = nn.Linear(hidden_dim * 5, hidden_dim, bias=False)
        self.m1 = nn.Linear(hidden_dim * 3, pooling_size * hidden_dim)
        self.m2 = nn.Linear(hidden_dim, pooling_size * hidden_dim)
        self.final = nn.Linear(hidden_dim * 2, pooling_size)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, U, cur_h, prev_start, prev_end):
        # U is of shape b, m, 2l, where b is batch size, 
        # m is number of words in the document,
        # and l is the hidden_dim.
        _, m, _ = U.size()
        everyone = torch.cat((cur_h, U[:,prev_start, :], U[:, prev_end, :]), -1)
        # everyone is now of shape b, 5l
        r = F.tanh(self.r(everyone))
        # r is now of shape b, l
        r_dup = r.unsqueeze(1).repeat(1, m, 1)
        # r_dup is now of shape b, m, l
        m1 = self.m1(torch.cat((U, r_dup), -1))
        # b, m, 3l times 3l, p*l = b, m, p*l. We would like to take max
        m1 = m1.view(-1, self.pooling_size, self.hidden_dim).max(2)
        # now m1 is b, m, l
        m2 = self.m2(m1)
        # m2 is now b, m, p*l, and we would like to take max
        m2 = m2.view(-1, self.pooling_size, self.hidden_dim).max(2)
        # m2 is now b, m, l
        alpha = self.final(torch.cat((m1, m2), -1))
        # alpha is now b, m, p, and we would like to take max
        alpha = alpha.max(-1)
        # Computation is done.

        return alpha