import torch
from torch import nn
from torch.nn.modules.linear import Linear
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.normalization import LayerNorm
import math
import numpy as np
from src.RWKV import Block, RWKV_Init
from src import DLinear_v10
 


# from DLinear import DLinear, DLinear_Init, Trend_Decompose

class SAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.temperature = math.sqrt(self.d_model / nhead)

        self.qtrans = nn.Linear(d_model, d_model, bias=False)
        self.ktrans = nn.Linear(d_model, d_model, bias=False)
        self.vtrans = nn.Linear(d_model, d_model, bias=False)

        attn_dropout_layer = []
        for i in range(nhead):
            attn_dropout_layer.append(Dropout(p=dropout))
        self.attn_dropout = nn.ModuleList(attn_dropout_layer)

        # input LayerNorm
        self.norm1 = LayerNorm(d_model, eps=1e-5)

        # FFN layerNorm
        self.norm2 = LayerNorm(d_model, eps=1e-5)
        self.ffn = nn.Sequential(
            Linear(d_model, d_model),
            nn.ReLU(),
            Dropout(p=dropout),
            Linear(d_model, d_model),
            Dropout(p=dropout)
        )

    def forward(self, x):
        x = self.norm1(x)
        q = self.qtrans(x).transpose(0, 1)
        k = self.ktrans(x).transpose(0, 1)
        v = self.vtrans(x).transpose(0, 1)
        # print("q,k,v",q.shape,k.shape,v.shape)

        dim = int(self.d_model / self.nhead)
        att_output = []



        for i in range(self.nhead):
            if i == self.nhead - 1:
                qh = q[:, :, i * dim:]
                kh = k[:, :, i * dim:]
                vh = v[:, :, i * dim:]
            else:
                qh = q[:, :, i * dim:(i + 1) * dim]
                kh = k[:, :, i * dim:(i + 1) * dim]
                vh = v[:, :, i * dim:(i + 1) * dim]

            atten_ave_matrixh = torch.softmax(torch.matmul(qh, kh.transpose(1, 2)) / self.temperature, dim=-1)
            if self.attn_dropout:
                atten_ave_matrixh = self.attn_dropout[i](atten_ave_matrixh)
            att_output.append(torch.matmul(atten_ave_matrixh, vh).transpose(0, 1))
        att_output = torch.concat(att_output, dim=-1)


        # FFN
        xt = x + att_output
        xt = self.norm2(xt)
        att_output = xt + self.ffn(xt)

        return att_output


class Filter(nn.Module):
    def __init__(self, d_input, d_output, seq_len, kernel=5, stride=5):
        super().__init__()
        self.d_input = d_input
        self.d_output = d_output
        self.seq_len = seq_len

        self.trans = nn.Linear(d_input, d_output)

        self.aggregate = nn.Conv1d(d_output, d_output, kernel_size=kernel, stride=stride, groups=d_output)

        # 输入是[N, T, d_feat]
        conv_feat = math.floor((self.seq_len - kernel) / stride + 1)

        self.proj_out = nn.Linear(conv_feat, 1)

    def forward(self, x):
        x = self.trans.forward(x)  # [N, T, d_feat]
        x_trans = x.transpose(-1, -2)  # [N, d_feat, T]
        x_agg = self.aggregate.forward(x_trans)  # [N, d_feat, conv_feat]
        out = self.proj_out.forward(x_agg)  # [N, d_feat, 1]
        return out.transpose(-1, -2)  # [N, 1, d_feat]


class TemporalAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.trans = nn.Linear(d_model, d_model, bias=False)

    def forward(self, z):
        # print("z.shape,",z.shape)  z.shape, torch.Size([256, 8, 256])
        h = self.trans(z)  # [N, T, D]
        # print("h.shape,",h.shape)    h.shape, torch.Size([256, 8, 256])
        query = h[:, -1, :].unsqueeze(-1)
        # print("query.shape,",query.shape)   query.shape, torch.Size([256, 256, 1])
        lam = torch.matmul(h, query).squeeze(-1)  # [N, T, D] --> [N, T]
        # print("lam.shape,",lam.shape)         lam.shape, torch.Size([256, 8])
        lam = torch.softmax(lam, dim=1).unsqueeze(1)
        # print("lamlam.shape,",lam.shape)         lamlam.shape, torch.Size([256, 1, 8])
        output = torch.matmul(lam, z).squeeze(1)  # [N, 1, T], [N, T, D] --> [N, 1, D]
        # print("output.shape,",output.shape)      output.shape, torch.Size([256, 256])
        return output


class DFT(nn.Module): 
    def __init__(self, d_feat=158, d_model=256, t_nhead=4, s_nhead=2,
                 seq_len=8, S_dropout_rate=0.5,
                 gate_input_start_index=158, gate_input_end_index=221, beta=None):
        super().__init__()

        self.d_feat = d_feat
        self.d_model = d_model
        self.n_attn = d_model
        self.n_head = t_nhead

        # market
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        self.d_gate_input = (gate_input_end_index - gate_input_start_index)  # F'
        self.feature_gate = Filter(self.d_gate_input, self.d_feat, seq_len)

        self.rwkv_trend = Block(layer_id=0, n_embd=self.d_model,
                                n_attn=self.n_attn, n_head=self.n_head, ctx_len=300,
                                n_ffn=self.d_model, hidden_sz=self.d_model)
        RWKV_Init(self.rwkv_trend, vocab_size=self.d_model, n_embd=self.d_model, rwkv_emb_scale=1.0)
        self.rwkv_season = Block(layer_id=0, n_embd=self.d_model,
                                 n_attn=self.n_attn, n_head=self.n_head, ctx_len=300,
                                 n_ffn=self.d_model, hidden_sz=self.d_model)
        RWKV_Init(self.rwkv_season, vocab_size=self.d_model, n_embd=self.d_model, rwkv_emb_scale=1.0)

        self.feat_to_model = nn.Linear(d_feat, d_model)  # 维度转化
        self.dlinear = DLinear_v10.DLinear(seq_len=seq_len, pred_len=seq_len, enc_in=self.d_model, kernel_size=3,
                                           individual=False)
        self.trend_TC = nn.Sequential(
            SAttention(d_model=d_model, nhead=s_nhead, dropout=S_dropout_rate),  # Stock correlation
            self.rwkv_trend  # Time correlation
        )
        self.season_TC = nn.Sequential(
            self.rwkv_season,  # Time correlation
            SAttention(d_model=d_model, nhead=s_nhead, dropout=S_dropout_rate),  # Stock correlation
        )
        self.out = nn.Sequential(
            TemporalAttention(d_model=d_model),
            # decoder
            nn.Linear(d_model, 1)
        )
        self.market_linear = nn.Linear(d_feat, d_model)

    def forward(self, x):
        src = x[:, :, :self.gate_input_start_index]  # N, T, D
        gate_input = x[:, :, self.gate_input_start_index:self.gate_input_end_index]
        market = self.feature_gate.forward(gate_input)

        src_model = self.feat_to_model(src)
        src_trend, src_season = self.dlinear(src_model)
        src_trend = self.trend_TC(src_trend) + self.market_linear(market)
        src_season = self.season_TC(src_season)

        output = self.out(src_trend + src_season).squeeze(-1)
        return output



if __name__ == "__main__":
    # x_sample = torch.randn((200, 221, 8))
    # conv = nn.Conv1d(221, 221, groups=221, kernel_size=5, stride=5)
    # x_conv = conv.forward(x_sample)
    # print(x_conv.shape) #torch.Size([200, 221, 1])

    # d = torch.randn((256, 8, 21))
    # gate = Filter(21, 158, 8)
    # out = gate.forward(d)
    # print(out.shape)    #torch.Size([256, 1, 158])

    input = torch.randn((1000, 8, 500))  
    v_first = input 
    # Batch Size/sample number样本数量, Feature Dimensions特征维度， Time Steps时间步数,
    print(input.shape)
    model = DFT()
    y = model.forward(input)
    print(y.shape)

    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    print("Total Parameters:", total_params)

    """
    x.shape torch.Size([188, 8, 221])
    src.shape torch.Size([188, 8, 158])
    gate_input.shape torch.Size([188, 8, 63])
    src111.shape torch.Size([188, 8, 256])
    src222.shape torch.Size([188, 8, 256])
    src333.shape torch.Size([188, 8, 256])
    src444.shape torch.Size([188, 8, 256])
    src555.shape torch.Size([188, 256])
    src666.shape torch.Size([188, 1])
    output.shape torch.Size([188])
    """
