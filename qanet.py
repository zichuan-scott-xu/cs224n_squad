import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import HighwayEncoder
import numpy as np


class DepthWiseSeparableConv(nn.Module):
    """DepthWise Seperable Convolutional Network Based on the paper and 
    repo and implementation from 
    https://github.com/heliumsea/QANet-pytorch/blob/master/models.py.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dim,):
        super().__init__()
        if dim == 1:
            self.depthwise_conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, 
                                            groups=in_channels, padding=kernel_size // 2)
            self.pointwise_conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0)
        elif dim == 2:
            self.depthwise_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, 
                                            groups=in_channels, padding=kernel_size // 2)
            self.pointwise_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0)
        nn.init.kaiming_normal_(self.depthwise_conv.weight)
        nn.init.constant_(self.depthwise_conv.bias, 0.0)
        nn.init.kaiming_normal_(self.depthwise_conv.weight)
        nn.init.constant_(self.pointwise_conv.bias, 0.0)

    def forward(self, x):
        return self.pointwise_conv(self.depthwise_conv(x))


class HighwayEncoder(nn.Module):
    """Encode an input sequence using a highway network.
    (Modified by transposing x to fit the dimension)

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
        #x : [b, word_dim + char_dim, seq_len]
        x = x.transpose(1, 2) # [b, seq_len, word_dim+char_dim=hidden_size]
        for gate, transform in zip(self.gates, self.transforms):
            # Shapes of g, t, and x are all (batch_size, seq_len, hidden_size)
            g = torch.sigmoid(gate(x))
            t = F.relu(transform(x))
            x = g * t + (1 - g) * x
        x = x.transpose(1, 2)
        return x



class Embedding(nn.Module):
    def __init__(self, word_dim, char_dim, drop_prob_word=0.1, drop_prob_char=0.05):
        super().__init__()
        self.conv2d = DepthWiseSeparableConv(char_dim, char_dim, 5, dim=2)
        self.highway = HighwayEncoder(2, word_dim + char_dim)
        self.drop_prob_word = drop_prob_word
        self.drop_prob_char = drop_prob_char

    def forward(self, ch_emb, wd_emb):
        wd_emb = F.dropout(wd_emb, p=self.drop_prob_word, training=self.training)
        wd_emb = wd_emb.transpose(1, 2) # [b, word_dim, c_len or q_len]

        ch_emb = ch_emb.permute(0, 3, 1, 2) # [b, char_dim, cq_len, char_limit]
        ch_emb = F.dropout(ch_emb, p=self.drop_prob_char, training=self.training)
        ch_emb = self.conv2d(ch_emb) # [b, char_dim, cq_len, char_limit]
        ch_emb = F.relu(ch_emb) # [b, char_dim, cq_len, char_limit]
        ch_emb, _ = torch.max(ch_emb, -1)  # [b, char_dim, cq_len]
        emb = torch.cat([ch_emb, wd_emb], dim=1) # [b, word_dim + char_dim, cq_len]
        emb = self.highway(emb) # [b, word_dim + char_dim, cq_len]
        return emb


class CoAttention():
    def __init__(self):
        pass


class SelfAttention():
    def __init__(self):
        pass



class PositionEncoder():
    def __init__(self, dim, length):
        # PE[pos, 2i] = sin(pos/10000^{2i/dim})
        # PE[pos, 2i+1] = cos(pos/10000^{2i/dim})
        self.PE = torch.empty((dim, length))
        for pos in dim:
            for j in length:
                if j % 2 == 0:
                    self.PE[pos, j] = np.sin(pos/10000**(j/dim))
                else:
                    self.PE[pos, j] = np.sin(pos/10000**((j-1)/dim))

    def forward(self, x):
        return x + self.PE


class QANetEncoderBlock(nn.Module):
    def __init__(self, d, max_length, num_conv=4, kernel_size=7, layer_dropout=0.1):
        super().__init__()
        self.num_conv = num_conv
        self.layer_dropout = layer_dropout
        self.pos_enc = PositionEncoder(d, max_length)
        self.conv_re = nn.ModuleList([DepthWiseSeparableConv(d, d, kernel_size=kernel_size) for _ in num_conv])
        self.layernorm_re = nn.ModuleList([nn.LayerNorm([d, max_length]) for _ in range(num_conv)])
        self.att = SelfAttention() # TODO
        self.layernorm2 = nn.LayerNorm([d, max_length])
        self.layernorm3 = nn.LayerNorm([d, max_length])
        self.fc = nn.Linear(in_features=d, out_features=d)
        

    def forward(self, x, mask):
        x = self.pos_enc(x)
        prevx = x
        for i in range(self.num_conv):
            x = self.layernorm_re[i](x)
            x = F.relu(self.conv_re[i](x))
            x = x + prevx
            if i % 2 == 1:
                drop_prob = self.layer_dropout * (i + 1) / self.num_conv
                x = F.dropout(x, p=drop_prob, training=self.training)
            prevx = x
        x = self.layernorm2(x)
        x = self.att(x)
        x = x + prevx
        x = F.dropout(x, self.layer_dropout, training=self.training)
        prevx = x
        x = self.layernorm3(x)
        x = F.relu(self.fc(x.transpose(1,2)).transpose(1.2))
        x = prevx + x
        x = F.dropout(x, self.layer_dropout, training=self.training)
        return x




class QANet(nn.Module):
    def __init__(self, word_vectors, char_vectors):
        super().__init__()
        self.word_dim = word_vectors.shape[1]
        self.char_dim = char_vectors.shape[1]
        # ------ Layers -------
        self.char_emb = nn.Embedding.from_pretrained(torch.Tensor(char_vectors))
        self.word_emb = nn.Embedding.from_pretrained(torch.Tensor(word_vectors))
        self.emb = Embedding(self.word_dim, self.char_dim)
        self.c_emb_encoder = QANetEncoderBlock(d=128, max_length=400)
        self.q_emb_encoder = QANetEncoderBlock(d=128, max_length=50)

    def forward(self, cw_idxs, cc_idxs, qw_idxs, qc_idxs):
        c_mask = (torch.zeros_like(cw_idxs) != cw_idxs).float()
        q_mask = (torch.zeros_like(qw_idxs) != qw_idxs).float()
        Cw = self.word_emb(cw_idxs) # [b, 246, 300] = [b, c_len, emb_dim]
        Qw = self.word_emb(qw_idxs) # [b, 23, 300]  = [b, q_len, emb_dim]
        # char_limit = maximum character kept from a word
        Cc = self.char_emb(cc_idxs) # [b, 246, 16, 64] = [b, c_len, char_limit, char_dim]
        Qc = self.char_emb(qc_idxs) # [b, 23, 16, 64]  = [b, q_len, char_limit, char_dim]
        C = self.emb(Cc, Cw) # [b, word_dim+char_dim, c_len]#
        Q = self.emb(Qc, Qw) # [b, word_dim+char_dim, q_len]