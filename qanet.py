import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


def mask_logits(x, mask):
    return x * mask + (-1e15) * (1 - mask)

class DepthWiseSeparableConv(nn.Module):
    """DepthWise Seperable Convolutional Network Based on the paper and 
    repo and implementation from 
    https://github.com/heliumsea/QANet-pytorch/blob/master/models.py.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dim=1):
        super().__init__()
        if dim == 1:
            self.depthwise_conv = nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, 
                                            groups=in_channels, padding=kernel_size // 2)
            self.pointwise_conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0)
        elif dim == 2:
            self.depthwise_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, 
                                            groups=in_channels, padding=kernel_size // 2)
            self.pointwise_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0)
        nn.init.kaiming_normal_(self.depthwise_conv.weight)
        nn.init.constant_(self.depthwise_conv.bias, 0.0)
        nn.init.kaiming_normal_(self.depthwise_conv.weight)
        nn.init.constant_(self.pointwise_conv.bias, 0.0)

    def forward(self, x):
        return self.pointwise_conv(self.depthwise_conv(x))


class HighwayEncoder(nn.Module):
    def __init__(self, num_layers, hidden_size):
        super().__init__()
        self.num_layers = num_layers
        self.linear = nn.ModuleList([Initialized_Conv1d(hidden_size, hidden_size, relu=False, bias=True) for _ in range(num_layers)])
        self.gate = nn.ModuleList([Initialized_Conv1d(hidden_size, hidden_size, bias=True) for _ in range(num_layers)])

    def forward(self, x):
        #x: shape [batch_size, hidden_size, length]
        dropout = 0.1
        for i in range(self.num_layers):
            gate = F.sigmoid(self.gate[i](x))
            nonlinear = self.linear[i](x)
            nonlinear = F.dropout(nonlinear, p=dropout, training=self.training)
            x = gate * nonlinear + (1 - gate) * x
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


class CoAttention(nn.Module):
    def __init__(self, model_dim, dropout_prob):
        super().__init__()
        self.model_dim = model_dim
        self.dropout_prob = dropout_prob
        W0 = torch.empty(3 * model_dim)
        nn.init.uniform_(W0, -1 / math.sqrt(model_dim), 1 / math.sqrt(model_dim))
        self.W0 = nn.Parameter(W0)
        # self.resizer = DepthWiseSeparableConv(model_dim * 4, model_dim, 5)
        self.resizer = Initialized_Conv1d(model_dim * 4, model_dim)

    def forward(self, C, Q, c_mask, q_mask):
        c_len, q_len = C.size(2), Q.size(2)
        C_t = C.transpose(1, 2) # [b, c_len, model_dim]
        Q_t = Q.transpose(1, 2) # [b, q_len, model_dim]
        C = C_t.unsqueeze(2).expand(-1, -1, q_len, -1) # [b, c_len, model_dim] -> [b, c_len, q_len, model_dim]
        Q = Q_t.unsqueeze(1).expand(-1, c_len, -1, -1) # [b, q_len, model_dim] -> [b, c_len, q_len, model_dim]
        c_mask = c_mask.unsqueeze(-1)# [b, c_len, 1]
        q_mask = q_mask.unsqueeze(1) # [b, 1, q_len]
        C_Q = torch.mul(C, Q)
        S = torch.cat([C, Q, C_Q], dim=3) # [b, c_len, q_len, 3 * model_dim]
        S = torch.matmul(S, self.W0) # [b, c_len, q_len]
        S_bar = F.softmax(mask_logits(S, q_mask), dim=2) # [b, c_len, q_len]
        S_bbar = F.softmax(mask_logits(S, c_mask), dim=1) #[b, c_len, q_len]
        A = torch.bmm(S_bar, Q_t) # [b, c_len, model_dim] 
        B = torch.bmm(torch.bmm(S_bar, S_bbar.transpose(1, 2)), C_t) # [b, c_len, model_dim] 
        out = torch.cat([C_t, A, torch.mul(C_t, A), torch.mul(C_t, B)], dim=2) # [b, c_len, 4 * model_dim]
        out = F.dropout(out, p=self.dropout_prob, training=self.training)
        out = out.transpose(1,2) # [b, 4*model_dim, c_len]
        out = self.resizer(out)
        return out


class Initialized_Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=1, stride=1, padding=0, groups=1,
                 relu=False, bias=False):
        super().__init__()
        self.out = nn.Conv1d(
            in_channels, out_channels,
            kernel_size, stride=stride,
            padding=padding, groups=groups, bias=bias)
        if relu is True:
            self.relu = True
            nn.init.kaiming_normal_(self.out.weight, nonlinearity='relu')
        else:
            self.relu = False
            nn.init.xavier_uniform_(self.out.weight)

    def forward(self, x):
        if self.relu is True:
            return F.relu(self.out(x))
        else:
            return self.out(x)


# TODO: modify the code for self attention
class SelfAttention(nn.Module):
    def __init__(self, model_dim, num_head, dropout):
        super().__init__()
        self.model_dim = model_dim
        self.num_head = num_head
        self.dropout = dropout
        self.mem_conv = Initialized_Conv1d(in_channels=model_dim, out_channels=model_dim*2, kernel_size=1, relu=False, bias=False)
        self.query_conv = Initialized_Conv1d(in_channels=model_dim, out_channels=model_dim, kernel_size=1, relu=False, bias=False)
        bias = torch.empty(1)
        nn.init.constant_(bias, 0)
        self.bias = nn.Parameter(bias)

    def forward(self, queries, mask):
        memory = queries
        memory = self.mem_conv(memory)
        query = self.query_conv(queries)
        memory = memory.transpose(1, 2)
        query = query.transpose(1, 2)
        Q = self.split_last_dim(query, self.num_head)
        K, V = [self.split_last_dim(tensor, self.num_head) for tensor in torch.split(memory, self.model_dim, dim=2)]

        key_depth_per_head = self.model_dim // self.num_head
        Q *= key_depth_per_head ** (-0.5)
        x = self.dot_product_attention(Q, K, V, mask = mask)
        return self.combine_last_two_dim(x.permute(0, 2, 1, 3)).transpose(1, 2)

    def dot_product_attention(self, q, k ,v, bias = False, mask = None):
        """dot-product attention.
        Args:
        q: a Tensor with shape [batch, heads, length_q, depth_k]
        k: a Tensor with shape [batch, heads, length_kv, depth_k]
        v: a Tensor with shape [batch, heads, length_kv, depth_v]
        bias: bias Tensor (see attention_bias())
        is_training: a bool of training
        scope: an optional string
        Returns:
        A Tensor.
        """
        logits = torch.matmul(q,k.permute(0,1,3,2))
        if bias:
            logits += self.bias
        if mask is not None:
            shapes = [x  if x != None else -1 for x in list(logits.size())]
            mask = mask.view(shapes[0], 1, 1, shapes[-1])
            logits = mask_logits(logits, mask)
        weights = F.softmax(logits, dim=-1)
        # dropping out the attention links for each of the heads
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        return torch.matmul(weights, v)

    def split_last_dim(self, x, n):
        """Reshape x so that the last dimension becomes two dimensions.
        The first of these two dimensions is n.
        Args:
        x: a Tensor with shape [..., m]
        n: an integer.
        Returns:
        a Tensor with shape [..., n, m/n]
        """
        old_shape = list(x.size())
        last = old_shape[-1]
        new_shape = old_shape[:-1] + [n] + [last // n if last else None]
        ret = x.view(new_shape)
        return ret.permute(0, 2, 1, 3)

    def combine_last_two_dim(self, x):
        """Reshape x so that the last two dimension become one.
        Args:
        x: a Tensor with shape [..., a, b]
        Returns:
        a Tensor with shape [..., ab]
        """
        old_shape = list(x.size())
        a, b = old_shape[-2:]
        new_shape = old_shape[:-2] + [a * b if a and b else None]
        ret = x.contiguous().view(new_shape)
        return ret



class MultiHeadAttention(nn.Module):
    def __init__(self, dim_model, num_heads, dropout_prob):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = dim_model // num_heads
        self.scaling = 1 / self.d_k ** 0.5

        self.q_proj = nn.Linear(dim_model, dim_model)
        self.v_proj = nn.Linear(dim_model, dim_model)
        self.k_proj = nn.Linear(dim_model, dim_model)
        self.fc = nn.Linear(dim_model, dim_model)
        self.dropout = nn.Dropout(dropout_prob)
    
    def forward(self, x, mask):
        b, model_dim, seq_len = x.size() # [b, model_dim, seq_len]
        x = x.transpose(1,2)
        Q = self.q_proj(x).view(b, seq_len, self.num_heads, self.d_k)
        Q = Q.permute(2, 0, 1, 3).contiguous().view(b * self.num_heads, seq_len, self.d_k)
        V = self.v_proj(x).view(b, seq_len, self.num_heads, self.d_k)
        V = V.permute(2, 0, 1, 3).contiguous().view(b * self.num_heads, seq_len, self.d_k)
        K = self.k_proj(x).view(b, seq_len, self.num_heads, self.d_k)
        K = K.permute(2, 0, 1, 3).contiguous().view(b * self.num_heads, seq_len, self.d_k)
        # [b, seq_len] -> [b, 1, seq_len] -> [b, seq_len, seq_len] -> [b*num_heads, seq_len, seq_len]
        mask = mask.unsqueeze(1).expand(-1, seq_len, -1).repeat(self.num_heads, 1, 1)    
        attn = torch.bmm(Q, K.transpose(1, 2)) * self.scaling # [b * num_heads, seq_len, seq_len]
        attn = mask_logits(attn, mask)
        attn = self.dropout(F.softmax(attn, dim=-1))
        attn = torch.bmm(attn, V) # [b * num_heads, seq_len, d_k]
        attn = attn.view(self.num_heads, b, seq_len, self.d_k).permute(1, 2, 0, 3).contiguous().view(b, seq_len, model_dim)
        out = self.dropout(self.fc(attn))
        out = out.transpose(1, 2) # [b, model_dim, seq_len]
        return out


class PositionEncoder(nn.Module):
    def __init__(self, model_dim, length):
        super().__init__()
        # PE[pos, 2i] = sin(pos/10000^{2i/model_dim})
        # PE[pos, 2i+1] = cos(pos/10000^{2i/model_dim})
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.PE = torch.empty((model_dim, length + 1)).to(self.device)
        for pos in range(model_dim):
            for j in range(length + 1):
                if j % 2 == 0:
                    self.PE[pos, j] = np.sin(pos/10000**(j/model_dim))
                else:
                    self.PE[pos, j] = np.sin(pos/10000**((j-1)/model_dim))

    def forward(self, x):
        seq_len = x.shape[2]
        return x + (self.PE[:, :seq_len]) # [b, model_dim, seq_len]


class QANetEncoderBlock(nn.Module):
    def __init__(self, d, max_length=400, num_conv=4, kernel_size=7, num_heads=1, layer_dropout=0.1):
        super().__init__()
        self.num_conv = num_conv
        self.layer_dropout = layer_dropout
        self.pos_enc = PositionEncoder(d, max_length)
        self.conv_re = nn.ModuleList([DepthWiseSeparableConv(d, d, kernel_size=kernel_size, dim=1) for _ in range(num_conv)])
        self.layernorm_re = nn.ModuleList([nn.LayerNorm(d) for _ in range(num_conv)])
        # self.att = MultiHeadAttention(d, num_heads=num_heads, dropout_prob=layer_dropout)
        self.att = SelfAttention(d, num_heads, dropout=layer_dropout) # TODO: change to use self attention
        self.layernorm2 = nn.LayerNorm(d)
        self.layernorm3 = nn.LayerNorm(d)
        self.fc = nn.Linear(in_features=d, out_features=d)
        

    def forward(self, x, mask):
        x = self.pos_enc(x) # [b, model_dim, seq_len]
        prevx = x
        for i in range(self.num_conv):
            x = self.layernorm_re[i](x.transpose(1,2)).transpose(1,2)
            x = F.relu(self.conv_re[i](x))
            x = x + prevx
            if i % 2 == 1:
                drop_prob = self.layer_dropout * (i + 1) / self.num_conv
                x = F.dropout(x, p=drop_prob, training=self.training)
            prevx = x
        x = self.layernorm2(x.transpose(1,2)).transpose(1,2)
        x = self.att(x, mask) # [b, model_dim, seq_len]
        x = x + prevx
        x = F.dropout(x, self.layer_dropout, training=self.training)
        prevx = x
        x = self.layernorm3(x.transpose(1,2)).transpose(1,2)
        x = F.relu(self.fc(x.transpose(1,2)).transpose(1,2))
        x = prevx + x
        x = F.dropout(x, self.layer_dropout, training=self.training)
        return x # [b, model_dim, seq_len]


class QANetDecoder(nn.Module):
    def __init__(self, model_dim):
        super().__init__()
        W1 = torch.empty(model_dim * 2)
        W2 = torch.empty(model_dim * 2)
        nn.init.uniform_(W1, -1 / math.sqrt(model_dim), 1 / math.sqrt(model_dim))
        nn.init.uniform_(W2, -1 / math.sqrt(model_dim), 1 / math.sqrt(model_dim))
        self.W1 = nn.Parameter(W1)
        self.W2 = nn.Parameter(W2)

    def forward(self, M0, M1, M2, mask):
        X1 = torch.cat([M0, M1], dim=1)
        X2 = torch.cat([M0, M2], dim=1)
        Y1 = torch.matmul(self.W1, X1)
        Y2 = torch.matmul(self.W2, X2)
        Y1 = maskout_padding(Y1, mask)
        Y2 = maskout_padding(Y2, mask)
        p1 = F.log_softmax(Y1, dim=1)
        p2 = F.log_softmax(Y2, dim=1)
        return p1, p2



class QANet(nn.Module):
    def __init__(self, word_vectors, char_vectors, model_dim, num_heads, layer_dropout=0.1, num_model_enc_block=5):
        super().__init__()
        self.word_dim = word_vectors.shape[1]
        self.char_dim = char_vectors.shape[1]
        # ------ Layers -------
        self.char_emb = nn.Embedding.from_pretrained(torch.Tensor(char_vectors))
        self.word_emb = nn.Embedding.from_pretrained(torch.Tensor(word_vectors))
        self.emb = Embedding(self.word_dim, self.char_dim)
        self.c_emb_conv = DepthWiseSeparableConv(self.word_dim + self.char_dim, model_dim, 5)
        self.q_emb_conv = DepthWiseSeparableConv(self.word_dim + self.char_dim, model_dim, 5)
        self.c_emb_encoder = QANetEncoderBlock(d=model_dim, num_heads=num_heads, max_length=400)
        self.q_emb_encoder = QANetEncoderBlock(d=model_dim, num_heads=num_heads, max_length=50)
        self.coattention = CoAttention(model_dim, dropout_prob=layer_dropout)
        self.model_encs = nn.ModuleList([QANetEncoderBlock(d=model_dim, num_heads=num_heads, max_length=400, num_conv=2, kernel_size=5)] * num_model_enc_block)
        self.decoder = QANetDecoder(model_dim)

    def forward(self, cw_idxs, cc_idxs, qw_idxs, qc_idxs):
        c_mask = (torch.zeros_like(cw_idxs) != cw_idxs).float() # 1 if nonzero
        q_mask = (torch.zeros_like(qw_idxs) != qw_idxs).float()
        Cw = self.word_emb(cw_idxs) # [b, 246, 300] = [b, c_len, emb_dim]
        Qw = self.word_emb(qw_idxs) # [b, 23, 300]  = [b, q_len, emb_dim]

        ## Embedding phase
        # char_limit = maximum character kept from a word
        Cc = self.char_emb(cc_idxs) # [b, 246, 16, 64] = [b, c_len, char_limit, char_dim]
        Qc = self.char_emb(qc_idxs) # [b, 23, 16, 64]  = [b, q_len, char_limit, char_dim]
        C = self.emb(Cc, Cw) # [b, word_dim+char_dim, c_len]#
        Q = self.emb(Qc, Qw) # [b, word_dim+char_dim, q_len]
        C = self.c_emb_conv(C) # [b, model_dim, c_len]
        Q = self.q_emb_conv(Q) # [b, model_dim, q_len]

        ## Encoding phase
        C = self.c_emb_encoder(C, c_mask)
        Q = self.q_emb_encoder(Q, q_mask)
        M0 = self.coattention(C, Q, c_mask, q_mask)
        for model_enc in self.model_encs:
            M0 = model_enc(M0, c_mask)
        M1 = M0
        for model_enc in self.model_encs:
            M1 = model_enc(M1, c_mask)
        M2 = M1
        for model_enc in self.model_encs:
            M2 = model_enc(M2, c_mask)
        log_p1, log_p2 = self.decoder(M0, M1, M2, c_mask)
        return log_p1, log_p2