import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def mask_logits(target, mask):
    mask = mask.type(torch.float32)
    return target * mask + (1 - mask) * (-1e30)

'''
    Modified based on BangLiu's and Heliumsea's implementations
'''
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, k, bias=True):
        super().__init__()
        self.depthwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=in_ch, kernel_size=k, groups=in_ch, padding=k // 2, bias=False)
        self.pointwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, padding=0, bias=bias)
    def forward(self, x):
        return F.relu(self.pointwise_conv(self.depthwise_conv(x)))


'''
Adopted from BangLiu's implementation of QANet.
'''
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

class HighwayEncoder(nn.Module):
    def __init__(self, num_layers, hidden_size):
        super().__init__()
        self.num_layers = num_layers
        self.linear = nn.ModuleList([Initialized_Conv1d(hidden_size, hidden_size, relu=False, bias=True) for _ in range(num_layers)])
        self.gate = nn.ModuleList([Initialized_Conv1d(hidden_size, hidden_size, bias=True) for _ in range(num_layers)])

    def forward(self, x):
        dropout = 0.1
        for i in range(self.num_layers):
            gate = F.sigmoid(self.gate[i](x))
            nonlinear = self.linear[i](x)
            nonlinear = F.dropout(nonlinear, p=dropout, training=self.training)
            x = gate * nonlinear + (1 - gate) * x
        return x

class Embedding(nn.Module):
    def __init__(self, word_dim, char_dim, model_dim, drop_prob_word=0.1, drop_prob_char=0.05):
        super().__init__()
        self.conv_char = nn.Conv2d(char_dim, model_dim, kernel_size = (1,5), padding=0, bias=True)
        self.conv_emb = Initialized_Conv1d(word_dim + model_dim, model_dim, bias=False)
        self.highway = HighwayEncoder(2, model_dim)
        self.drop_prob_word = drop_prob_word
        self.drop_prob_char = drop_prob_char

    def forward(self, ch_emb, wd_emb):
        wd_emb = F.dropout(wd_emb, p=self.drop_prob_word, training=self.training)
        wd_emb = wd_emb.transpose(1, 2) # [b, word_dim, c_len or q_len]

        ch_emb = ch_emb.permute(0, 3, 1, 2) # [b, char_dim, cq_len, char_limit]
        ch_emb = F.dropout(ch_emb, p=self.drop_prob_char, training=self.training)
        ch_emb = self.conv_char(ch_emb) # [b, char_dim, cq_len, char_limit]
        ch_emb = F.relu(ch_emb) # [b, char_dim, cq_len, char_limit]
        ch_emb, _ = torch.max(ch_emb, -1)  # [b, char_dim, cq_len]

        emb = torch.cat([ch_emb, wd_emb], dim=1) # [b, word_dim + char_dim, cq_len]
        emb = self.conv_emb(emb)
        emb = self.highway(emb) # [b, word_dim + char_dim, cq_len]
        return emb

'''
Adopted from BangLiu's implementation of Positional Encoder.
'''
def PosEncoder(x, min_timescale=1.0, max_timescale=1.0e4):
    x = x.transpose(1, 2)
    length = x.size()[1]
    channels = x.size()[2]
    signal = get_timing_signal(length, channels, min_timescale, max_timescale)
    if x.get_device() < 0:
        return (x + signal.to('cpu')).transpose(1,2)
    else:
        return (x + signal.to(x.get_device())).transpose(1,2)


def get_timing_signal(length, channels,
                      min_timescale=1.0, max_timescale=1.0e4):
    position = torch.arange(length).type(torch.float32)
    num_timescales = channels // 2
    log_timescale_increment = (math.log(float(max_timescale) / float(min_timescale)) / (float(num_timescales) - 1))
    inv_timescales = min_timescale * torch.exp(
            torch.arange(num_timescales).type(torch.float32) * -log_timescale_increment)
    scaled_time = position.unsqueeze(1) * inv_timescales.unsqueeze(0)
    signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim = 1)
    m = nn.ZeroPad2d((0, (channels % 2), 0, 0))
    signal = m(signal)
    signal = signal.view(1, length, channels)
    return signal


class SelfAttention(nn.Module):
    def __init__(self, model_dim, num_heads, dropout):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.mem_conv = Initialized_Conv1d(in_channels=model_dim, out_channels=model_dim*2, kernel_size=1, relu=False, bias=False)
        self.query_conv = Initialized_Conv1d(in_channels=model_dim, out_channels=model_dim, kernel_size=1, relu=False, bias=False)

    def forward(self, x, mask):
        b, model_dim, seq_len = x.size() # [b, model_dim, seq_len] 

        memory = x
        memory = self.mem_conv(memory)  # [b, 2 * model_dim, seq_len]
        memory = memory.transpose(1, 2) # [b, seq_len, 2 * model_dim]
        x = self.query_conv(x)      # [b, model_dim, seq_len]
        Q = x.transpose(1, 2).view(b, seq_len, self.num_heads, model_dim // self.num_heads)   # [b, seq_len, num_h, model_dim // num_h]
        Q = Q.permute(0, 2, 1, 3) # [b, num_heads, seq_len, model_dim // num_h]
        K, V = [x.view(b, seq_len, self.num_heads, model_dim // self.num_heads).permute(0,2,1,3) for x in torch.split(memory, self.model_dim, dim=2)]
        # [b, num_heads, seq_len, model_dim // num_h]

        Q = Q * (self.model_dim // self.num_heads) ** -0.5

        logits = torch.matmul(Q, K.permute(0,1,3,2)) # [b, num_heads, seq_len, seq_len]
        mask = mask.view(b, 1, 1, seq_len)
        logits = mask_logits(logits, mask)
        alpha = F.softmax(logits, dim=-1) # [b, num_heads, seq_len, seq_len]
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        attn = torch.matmul(alpha, V) # [b, num_heads, seq_len, model_dim // num_h]
        attn = attn.permute(0,2,1,3) # [b, seq_len, num_heads, model_dim // num_h]
        attn = attn.reshape(b, seq_len, model_dim).transpose(1,2) # [b, model_dim, seq_len]
        
        return attn
    

class QANetEncoderBlock(nn.Module):
    def __init__(self, model_dim, max_length=400, num_conv=4, kernel_size=7, num_heads=1, layer_dropout=0.1):
        super().__init__()
        self.num_conv = num_conv
        self.dropout = layer_dropout
        self.conv_re = nn.ModuleList([DepthwiseSeparableConv(model_dim, model_dim, k=kernel_size) for _ in range(num_conv)])
        self.layernorm_re = nn.ModuleList([nn.LayerNorm(model_dim) for _ in range(num_conv)])
        # self.att = MultiHeadAttention(d, num_heads=num_heads, dropout_prob=layer_dropout)
        self.att = SelfAttention(model_dim, num_heads, dropout=layer_dropout)

        # Layernorms after convolutional blocks
        self.layernorm2 = nn.LayerNorm(model_dim)
        self.layernorm3 = nn.LayerNorm(model_dim)

        # feedfoward layer after attentions
        self.ff1 = Initialized_Conv1d(model_dim, model_dim, relu=True, bias=True)
        self.ff2 = Initialized_Conv1d(model_dim, model_dim, bias=True)
        

    def forward(self, x, mask, l, blks):
        total_layers = (self.num_conv + 1) * blks
        out = PosEncoder(x) # [b, model_dim, seq_len]
        for i, conv in enumerate(self.conv_re):
            res = out
            out = self.layernorm_re[i](out.transpose(1,2)).transpose(1,2)
            if i % 2 == 0:
                out = F.dropout(out, p=self.dropout, training=self.training)
            out = conv(out)
            out = self.layer_dropout(out, res, self.dropout * float(l) / total_layers)
            l += 1
        res = out
        out = self.layernorm2(out.transpose(1,2)).transpose(1,2)
        out = F.dropout(out, p=self.dropout, training=self.training)
        out = self.att(out, mask) # [b, model_dim, seq_len]
        out = self.layer_dropout(out, res, self.dropout*float(l) / total_layers)
        l += 1
        res = out

        out = self.layernorm3(out.transpose(1,2)).transpose(1,2)
        out = F.dropout(out, p=self.dropout, training=self.training)
        out = self.ff2(self.ff1(out))
        out = self.layer_dropout(out, res, self.dropout * float(l) / total_layers)
        return out


    def layer_dropout(self, inputs, residual, dropout):
            if self.training == True:
                pred = torch.empty(1).uniform_(0,1) < dropout
                if pred:
                    return residual
                else:
                    return F.dropout(inputs, dropout, training=self.training) + residual
            else:
                return inputs + residual


class CoAttention(nn.Module):
    def __init__(self, model_dim, dropout_prob):
        super().__init__()
        # self.model_dim = model_dim
        # self.dropout_prob = dropout_prob
        # W0 = torch.empty(3 * model_dim)
        # nn.init.uniform_(W0, -1 / math.sqrt(model_dim), 1 / math.sqrt(model_dim))
        # self.W0 = nn.Parameter(W0)
        # # self.resizer = DepthWiseSeparableConv(model_dim * 4, model_dim, 5)
        # self.resizer = Initialized_Conv1d(model_dim * 4, model_dim)
        w4C = torch.empty(model_dim, 1)
        w4Q = torch.empty(model_dim, 1)
        w4mlu = torch.empty(1, 1, model_dim)
        nn.init.xavier_uniform_(w4C)
        nn.init.xavier_uniform_(w4Q)
        nn.init.xavier_uniform_(w4mlu)
        self.w4C = nn.Parameter(w4C)
        self.w4Q = nn.Parameter(w4Q)
        self.w4mlu = nn.Parameter(w4mlu)

        bias = torch.empty(1)
        nn.init.constant_(bias, 0)
        self.bias = nn.Parameter(bias)
        self.dropout = dropout_prob

    def forward(self, C, Q, c_mask, q_mask):
        # c_len, q_len = C.size(2), Q.size(2)
        # C_t = C.transpose(1, 2) # [b, c_len, model_dim]
        # Q_t = Q.transpose(1, 2) # [b, q_len, model_dim]
        # C = C_t.unsqueeze(2).expand(-1, -1, q_len, -1) # [b, c_len, model_dim] -> [b, c_len, q_len, model_dim]
        # Q = Q_t.unsqueeze(1).expand(-1, c_len, -1, -1) # [b, q_len, model_dim] -> [b, c_len, q_len, model_dim]
        # c_mask = c_mask.unsqueeze(-1)# [b, c_len, 1]
        # q_mask = q_mask.unsqueeze(1) # [b, 1, q_len]
        # C_Q = torch.mul(C, Q)
        # S = torch.cat([C, Q, C_Q], dim=3) # [b, c_len, q_len, 3 * model_dim]
        # S = torch.matmul(S, self.W0) # [b, c_len, q_len]
        # S_bar = F.softmax(mask_logits(S, q_mask), dim=2) # [b, c_len, q_len]
        # S_bbar = F.softmax(mask_logits(S, c_mask), dim=1) #[b, c_len, q_len]
        # A = torch.bmm(S_bar, Q_t) # [b, c_len, model_dim] 
        # B = torch.bmm(torch.bmm(S_bar, S_bbar.transpose(1, 2)), C_t) # [b, c_len, model_dim] 
        # out = torch.cat([C_t, A, torch.mul(C_t, A), torch.mul(C_t, B)], dim=2) # [b, c_len, 4 * model_dim]
        # out = F.dropout(out, p=self.dropout_prob, training=self.training)
        # out = out.transpose(1,2) # [b, 4*model_dim, c_len]
        # out = self.resizer(out)
        # return out
        C = C.transpose(1, 2)
        Q = Q.transpose(1, 2)
        batch_size_c = C.size()[0]
        batch_size, Lc, d_model = C.shape
        batch_size, Lq, d_model = Q.shape
        S = self.trilinear_for_attention(C, Q)
        c_mask = c_mask.view(batch_size_c, Lc, 1)
        q_mask = q_mask.view(batch_size_c, 1, Lq)
        S1 = F.softmax(mask_logits(S, q_mask), dim=2)
        S2 = F.softmax(mask_logits(S, c_mask), dim=1)
        A = torch.bmm(S1, Q)
        B = torch.bmm(torch.bmm(S1, S2.transpose(1, 2)), C)
        out = torch.cat([C, A, torch.mul(C, A), torch.mul(C, B)], dim=2)
        return out.transpose(1, 2)

    '''
    Adopted from BangLiu's implementation. We test over our own implementation and BangLiu's implementation 
    and found similar performance. 
    '''
    def trilinear_for_attention(self, C, Q):
        batch_size, Lc, d_model = C.shape
        batch_size, Lq, d_model = Q.shape
        dropout = self.dropout
        C = F.dropout(C, p=dropout, training=self.training)
        Q = F.dropout(Q, p=dropout, training=self.training)
        subres0 = torch.matmul(C, self.w4C).expand([-1, -1, Lq])
        subres1 = torch.matmul(Q, self.w4Q).transpose(1, 2).expand([-1, Lc, -1])
        subres2 = torch.matmul(C * self.w4mlu, Q.transpose(1,2))
        res = subres0 + subres1 + subres2
        res += self.bias
        return res

class QANetDecoder(nn.Module):
    def __init__(self, model_dim):
        super().__init__()
        self.w1 = Initialized_Conv1d(model_dim * 2, 1)
        self.w2 = Initialized_Conv1d(model_dim * 2, 1)

    def forward(self, M1, M2, M3, mask):
        X1 = torch.cat([M1, M2], dim=1)
        X2 = torch.cat([M1, M3], dim=1)
        Y1 = mask_logits(self.w1(X1).squeeze(), mask)
        Y2 = mask_logits(self.w2(X2).squeeze(), mask)
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
        self.emb = Embedding(self.word_dim, self.char_dim, model_dim)
        self.c_emb_encoder = QANetEncoderBlock(model_dim=model_dim, num_heads=num_heads, kernel_size=7, max_length=400)
        self.q_emb_encoder = QANetEncoderBlock(model_dim=model_dim, num_heads=num_heads, kernel_size=7, max_length=50)
        self.coattention = CoAttention(model_dim, dropout_prob=layer_dropout)
        self.model_encs = nn.ModuleList([QANetEncoderBlock(num_conv=2, model_dim=model_dim, num_heads=num_heads, kernel_size=5, layer_dropout=0.1) for _ in range(num_model_enc_block)])
        self.num_model_enc_block = 5
        self.decoder = QANetDecoder(model_dim)
        self.dropout = layer_dropout

    def forward(self, cw_idxs, cc_idxs, qw_idxs, qc_idxs):
        c_mask = (torch.zeros_like(cw_idxs) != cw_idxs).float() # 1 if nonzero
        q_mask = (torch.zeros_like(qw_idxs) != qw_idxs).float()
        Cw = self.word_emb(cw_idxs) # [b, 246, 300] = [b, c_len, emb_dim]
        Qw = self.word_emb(qw_idxs) # [b, 23, 300]  = [b, q_len, emb_dim]

        ## Embedding phase
        # char_limit = maximum character kept from a word
        Cc = self.char_emb(cc_idxs) 
        Qc = self.char_emb(qc_idxs) 
        C = self.emb(Cc, Cw) 
        Q = self.emb(Qc, Qw) 

        ## Encoding phase
        C = self.c_emb_encoder(C, c_mask, l=1, blks=1)
        Q = self.q_emb_encoder(Q, q_mask, l=1, blks=1)
        M0 = self.coattention(C, Q, c_mask, q_mask)
        for i, model_enc in enumerate(self.model_encs):
            M0 = model_enc(M0, c_mask, l=4*i+1, blks=self.num_model_enc_block)
        M1 = M0
        for i, model_enc in enumerate(self.model_encs):
            M0 = model_enc(M0, c_mask, l=4*i+1, blks=self.num_model_enc_block)
        M2 = M0
        M0 = F.dropout(M0, p=self.dropout, training=self.training)
        for i, model_enc in enumerate(self.model_encs):
            M0 = model_enc(M0, c_mask, l=4*i+1, blks=self.num_model_enc_block)
        M3 = M0
        log_p1, log_p2 = self.decoder(M1, M2, M3, c_mask)
        return log_p1, log_p2