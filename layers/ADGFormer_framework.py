__all__ = ['ADGFormer_framework']

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn

from layers.ADGFormer_layers import *
from layers.RevIN import RevIN

def check_nan_inf(tensor: Tensor, name: str = ""):
    if torch.isnan(tensor).any():
        print(f"NaN detected in {name}")
    if torch.isinf(tensor).any():
        print(f"Inf detected in {name}")
    else:
        print("ok"+name)


class ADGFormer_framework(nn.Module):
    def __init__(self, cossim_matrix: Tensor, n_vars: int, context_window: int, target_window: int, patch_len: int,
                 stride: int, device: str = 'cpu',num_adj_matrices: int = 24, d_graph=16, d_gcn=1, w_ratio=0.5, 
                 gat_layers: int = 2, gat_dropout: float = 0., bias_scale:float = 1.0,gat_tau:float=1.0,
                 s_mha_d_model: int = 64,s_mha_heads: int = 4, s_mha_d_k: Optional[int] = None,
                 gtu_stride: int = 1,
                 n_layers: int = 3, d_model=128, n_heads=16, d_k: Optional[int] = None, d_v: Optional[int] = None,
                 d_ff: int = 256, norm: str = 'BatchNorm', attn_dropout: float = 0., dropout: float = 0.,
                 act: str = "gelu", res_attention: bool = True, pre_norm: bool = False, store_attn: bool = False,
                 pe: str = 'zeros', learn_pe: bool = True, predictor_dropout=0, revin=True, affine=True, subtract_last=False,
                 verbose: bool = False, **kwargs):

        super().__init__()

        # RevIn
        self.revin = revin
        if self.revin: self.revin_layer = RevIN(n_vars, affine=affine, subtract_last=subtract_last)

        # Patching
        self.patch_len = patch_len
        self.stride = stride
        patch_num = int(context_window-patch_len ) // stride + 1
        if patch_num % 4 != 0:
            patch_num = patch_num+(4-patch_num%4)
        self.patch_num = patch_num
        # print(f"patch_num: {patch_num}")
        # print(f"patch_len: {patch_len}")

        # MTT
        self.mtt = MTT(patch_num=patch_num, patch_len=patch_len, n_layers=n_layers, d_model=d_model, n_heads=n_heads,
                       d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout, act=act,
                       res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn, pe=pe, learn_pe=learn_pe,
                       verbose=verbose, **kwargs)

        self.device = device
        self.d_graph = d_graph
        self.w_ratio = w_ratio
        # self.mp_layers = mp_layers
        self.cossim_matrix = cossim_matrix
        self.n_vars = n_vars
        self.gc = Graph_constructor(self.n_vars, self.d_graph, self.device, self.cossim_matrix,num_adj_matrices=num_adj_matrices, w_ratio=self.w_ratio)
        self.register_buffer("static_adj", self._prepare_static_adj(self.cossim_matrix))
    
        self.s_mha_d_model = s_mha_d_model
        self.s_mha_heads = s_mha_heads
        d_k_eff = (s_mha_d_model // s_mha_heads) if s_mha_d_k is None else s_mha_d_k
        self.node_embed = nn.Linear(1, s_mha_d_model, bias=False)   # 把每个时刻的标量温度 -> 节点嵌入
        self.spatial_mha = SMultiHeadAttention(d_model=s_mha_d_model, d_k=d_k_eff, n_heads=s_mha_heads)

        self.gat_dyn = SpatialWeightedGAT(in_dim=1, out_dim=1,
                                        heads=s_mha_heads, mp_layers=gat_layers,
                                        dropout=gat_dropout, merge='mean',
                                        use_log=False, tau=gat_tau, bias_scale=bias_scale)

        self.gat_sta = SpatialWeightedGAT(in_dim=1, out_dim=1,
                                        heads=s_mha_heads, mp_layers=gat_layers,
                                        dropout=gat_dropout, merge='mean',
                                        use_log=False, tau=gat_tau, bias_scale=bias_scale)

        self.fuse_gate = nn.Sequential(nn.Linear(2, 1), nn.Sigmoid())

        # Temporal convolution (GTU)
        self.gtu3 = GTU(in_channels=d_model * 4, time_strides=gtu_stride, kernel_size=3)
        self.gtu5 = GTU(in_channels=d_model * 4, time_strides=gtu_stride, kernel_size=5)    # 时间尺度 5
        self.gtu7 = GTU(in_channels=d_model * 4, time_strides=gtu_stride, kernel_size=7)    # 时间尺度 7
        self.fcmy = nn.Sequential(
            nn.Linear(int(3 * (patch_num / 4) - 12), int(patch_num / 4)),
            nn.Dropout(0.05),
        )
        # Predictor
        self.nf = d_model * patch_num
        self.n_vars = n_vars
        self.d_gcn = d_gcn
        self.pred = Predictor(self.n_vars, self.nf, target_window, predictor_dropout=predictor_dropout)

    def _prepare_static_adj(self, cossim_matrix: Tensor) -> Tensor:
        """
        根据预定义相似度矩阵构建固定邻接矩阵：
        - 去掉负值
        - 去掉自环 & 再加上单位阵
        - 行归一化
        """
        adj = F.relu(cossim_matrix.clone())
        adj = adj - torch.diag(torch.diag(adj))
        adj = adj + torch.eye(adj.size(0), device=adj.device)
        d = adj.sum(1)
        adj = adj / d.view(-1, 1)
        return adj  # [n_vars, n_vars
    
    def forward(self, z, time_index, current_epoch):  # z: [bs x nvars x seq_len] time_index: [bs x timeinfo]
        B, N, T = z.shape
        # norm
        if self.revin:
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, 'norm')
            z = z.permute(0, 2, 1)


        # Dynamic graph
        adj_dyn = self.gc(time_index, current_epoch)
        adj_sta = self.static_adj.unsqueeze(0).unsqueeze(0).expand(B, T, N, N).contiguous()

        # Spatial attention
        BT = B * T
        x_bt = z.permute(0, 2, 1).reshape(BT, N, 1)                 # [BT, N, 1]
        emb_bt = self.node_embed(x_bt)                              # [BT, N, d_model]
        adj_dyn_bt = adj_dyn.reshape(BT, N, N)
        adj_sta_bt = adj_sta.reshape(BT, N, N)

        
        # mask: True 
        mask_dyn = (adj_dyn_bt <= 0)
        # S-MHA 用动态图做 mask；bias 逐层共用）
        scores_dyn = self.spatial_mha(emb_bt, attn_mask=mask_dyn)   # [BT, Hm, N, N]

        # # GAT
        h_dyn = self.gat_dyn(x_bt, adj_dyn_bt, bias=scores_dyn)        # [BT,N,1]
        h_sta = self.gat_sta(x_bt, adj_sta_bt, bias=scores_dyn)        # [BT,N,1]
        

        gate = self.fuse_gate(torch.cat([h_dyn, h_sta], dim=-1))    # [BT, N, 1], σ
       
        h_fused = gate * h_dyn + (1. - gate) * h_sta                # [BT, N, 1]

        x_spatial = h_fused.view(B, T, N, 1).permute(0, 2, 1, 3).squeeze(-1)  # [B, N, T]
        # print("x_spatial",x_spatial.shape)

        # 时间
        # do patching
        cur_len = x_spatial.shape[-1]
        needed_len = (self.patch_num-1)*self.stride+self.patch_len
        pad_len = needed_len -cur_len
        if pad_len > 0:
            x_spatial = F.pad(x_spatial,(0,pad_len),mode = 'replicate')
        x_patch = x_spatial.unfold(dimension=-1, size=self.patch_len, step=self.stride)  
        x_patch = x_patch.permute(0, 1, 3, 2)  

        # MTT
        x_time = self.mtt(x_patch)  

        # GTU 
        x_time = x_time.permute(0, 2, 1, 3).contiguous()  
        x_gtu3 = self.gtu3(x_time)  
        x_gtu5 = self.gtu5(x_time)  
        x_gtu7 = self.gtu7(x_time)  
        x_time_multiscale = torch.cat([x_gtu3, x_gtu5, x_gtu7], dim=-1)  
        x_time = x_time_multiscale.permute(0, 2, 1, 3).contiguous()            


        x_time = self.fcmy(x_time)   # [B, N, C, T*]


        # Predictor
        z = self.pred(x_time)  # z: [bs x nvars x target_window]

        # denorm
        if self.revin:
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, 'denorm')
            z = z.permute(0, 2, 1)
        return z
        

class SScaledDotProductAttention(nn.Module):
    def __init__(self, d_k: int):
        super().__init__()
        self.d_k = d_k

    def forward(self, Q, K, attn_mask):
        # Q: [B, H, N, d_k], K: [B, H, N, d_k], mask: [B, N, N] 或 [B, 1, N, N]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)  # [B, H, N, N]
        if attn_mask is not None:
            if attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)  # -> [B, 1, N, N]
            scores = scores.to(torch.float32)  # 转换为 float32
            scores = scores.masked_fill(attn_mask, -1e9)
            # print(scores.max(),scores.min())
            # scores = scores.to(torch.float16)  # 如果后续需要使用 float16

        return scores
class SMultiHeadAttention(nn.Module):
    """
    仅做 QK^T 打分，返回 raw scores: [B, H, N, N]
    """
    def __init__(self, d_model: int, d_k: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.n_heads = n_heads
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.s_attn = SScaledDotProductAttention(d_k)

    def forward(self, x, attn_mask=None):
        # x: [B, N, d_model]
        B, N, _ = x.shape
        Q = self.W_Q(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)  # [B, H, N, d_k]
        K = self.W_K(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)  # [B, H, N, d_k]
        scores = self.s_attn(Q, K, attn_mask)  # [B, H, N, N]
        return scores

class WeightedGATWithBiasLayer(nn.Module):
    """
    单层：直接用邻接权作为基础打分，叠加 S-MHA 的 bias；不再计算 a_src/a_dst
    h     : [B, N, Fin]
    adj_w : [B, N, N] (>=0)
    bias  : [B, H, N, N] 或 [B, 1, N, N]
    返回  : [B, N, Fout]
    """
    def __init__(self, in_dim=1, out_dim=1, heads=4, dropout=0.,
                 merge='mean', use_log=True, tau=1.0, bias_scale=1.0, eps=1e-6):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.use_log = use_log
        self.tau = nn.Parameter(torch.tensor(float(tau)))
        self.bias_scale = nn.Parameter(torch.tensor(float(bias_scale)))
        self.eps = eps

        self.W = nn.Linear(in_dim, out_dim, bias=False)         # 值映射
        self.attn_dropout = nn.Dropout(dropout)

        if heads > 1 or out_dim > 1:
            if merge == 'mean':
                self.reduce = lambda x: x.mean(dim=-1, keepdim=True)  # [B,N,1]
            else:
                self.linear_reduce = nn.Linear(heads * out_dim, out_dim, bias=False)
                self.reduce = lambda x: self.linear_reduce(x)
        else:
            self.reduce = None

    def forward(self, h, adj_w, bias=None):
        B, N, _ = h.shape
        H, Fout = self.heads, self.out_dim
        V = self.W(h)                                        # [B,N,F]
        V = V.unsqueeze(1).expand(B, self.heads, N, self.out_dim)  # [B,H,N,F]

        # 基础打分来自图权
        if self.use_log:
            e = torch.log(adj_w + self.eps)                  # [B,N,N]
        else:
            e = adj_w
        e = self.tau * e
        e = e.unsqueeze(1).expand(B, self.heads, N, N)       # [B,H,N,N]

        # # 加 S-MHA 偏置
        # if bias is not None:
        #     if bias.dim() == 3:
        #         bias = bias.unsqueeze(1)
        #     e = e + self.bias_scale.view(1, 1, 1, 1) * bias                   # [B,H,N,N]

        # # 0 权位置屏蔽
        # mask = (adj_w <= 0).unsqueeze(1)                     # [B,1,N,N]
        # neg_inf = torch.full_like(e, -1e9)
        # e = torch.where(mask, neg_inf, e)

        # 0 权位置的 Mask 定义
        mask = (adj_w <= 0).unsqueeze(1)                     # [B,1,N,N]
        if bias is not None:
            if bias.dim() == 3:
                bias = bias.unsqueeze(1)
            
            # 计算加权后的 Bias
            bias_term = self.bias_scale.view(1, 1, 1, 1) * bias
            e =  torch.where(mask, bias_term, e+bias_term)
        else:
            neg_inf = torch.full_like(e, -1e9)
            e = torch.where(mask, neg_inf, e)

        # softmax 聚合
        attn = F.softmax(e, dim=-1)

        attn = self.attn_dropout(attn)                       # [B,H,N,N]
        h_prime = torch.matmul(attn, V)                      # [B,H,N,F]
        h_prime = h_prime.permute(0, 2, 1, 3).reshape(B, N, self.heads * self.out_dim)

        if self.reduce is not None:
            h_prime = self.reduce(h_prime)                   # [B,N,F] (默认 [B,N,1])
        # return F.elu(h_prime)
        attn_mean = attn.mean(dim=1)
        return F.elu(h_prime),attn_mean
    


class SpatialWeightedGAT(nn.Module):
    """
    多层堆叠的加权GAT（每层可残差），形状保持方便与时序模块对接
    """
    def __init__(self, in_dim=1, out_dim=1, heads=4, mp_layers=2, dropout=0.,
                 residual=True, **layer_kwargs):
        super().__init__()
        self.layers = nn.ModuleList([
            WeightedGATWithBiasLayer(in_dim=(in_dim if i == 0 else out_dim),
                                     out_dim=out_dim, heads=heads, dropout=dropout, **layer_kwargs)
            for i in range(mp_layers)
        ])
        self.residual = residual

    def forward(self, x, adj_w, bias=None):
        h = x
        last_attn = None
        for i,layer in enumerate(self.layers):
            h_new,attn= layer(h, adj_w, bias=bias)
            if self.residual and (h_new.shape == h.shape):
                h = h + h_new
            else:
                h = h_new
            if i == len(self.layers)-1:
                last_attn = attn
        return h,last_attn

    # def forward(self, x, adj_w, bias=None):
    #     h = x
    #     for layer in self.layers:
    #         h_new = layer(h, adj_w, bias=bias)
    #         if self.residual and (h_new.shape == h.shape):
    #             h = h + h_new
    #         else:
    #             h = h_new
    #     return h

class GTU(nn.Module):
    def __init__(self, in_channels, time_strides=1, kernel_size=1):
        super().__init__()
        self.in_channels = in_channels
        self.leaky_relu = F.leaky_relu
        self.sigmoid = nn.Sigmoid()
        self.con2out = nn.Conv2d(in_channels, 2 * in_channels,
                                 kernel_size=(1, kernel_size), stride=(1, time_strides))
    def forward(self, x):
        # x: [B, C, N, T]
        x_causal_conv = self.con2out(x)
        x_p = x_causal_conv[:, : self.in_channels, :, :]
        x_q = x_causal_conv[:, -self.in_channels:, :, :]
        x_gtu = torch.mul(self.leaky_relu(x_p), self.sigmoid(x_q))
        return x_gtu
    

class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('bnsc,bnms->bmsc', (x, A))
        return x.contiguous()


class Graph_constructor(nn.Module):
    def __init__(self, n_vars, d_graph, device, cossim_matrix, alpha=0.9, num_adj_matrices=7, w_ratio=0.5):
        super(Graph_constructor, self).__init__()

        self.n_vars = n_vars
        self.init_adj_matrix = cossim_matrix

        self.emb_list1 = nn.ParameterList(
            [nn.Parameter(torch.randn(n_vars, d_graph, device=device)) for _ in range(num_adj_matrices)])
        self.emb_list2 = nn.ParameterList(
            [nn.Parameter(torch.randn(n_vars, d_graph, device=device)) for _ in range(num_adj_matrices)])

        self.lin1 = nn.ModuleList([nn.Linear(d_graph, d_graph) for _ in range(num_adj_matrices)])
        self.lin2 = nn.ModuleList([nn.Linear(d_graph, d_graph) for _ in range(num_adj_matrices)])
        self.lin1.to(device)
        self.lin2.to(device)

        self.device = device
        self.alpha = alpha
        self.num_adj_matrices = num_adj_matrices
        self.w_ratio = w_ratio

    def forward(self, time_indices, current_epoch):
        adjs = []
        num_elements = self.n_vars * self.n_vars

        # Dynamic Multivariate Correlation Weight Learning
        for i in range(self.num_adj_matrices):
            nodevec1 = F.leaky_relu(self.lin1[i](self.emb_list1[i]))
            nodevec2 = F.leaky_relu(self.lin2[i](self.emb_list2[i]))
            # print("self.init_adj_matrix",self.init_adj_matrix.shape)
            # print(nodevec1.shape)
            # print(nodevec2.shape)

            prop = min(current_epoch / 5, self.alpha)
            a = (1 - prop) * self.init_adj_matrix + prop * torch.mm(nodevec1, nodevec2.transpose(1, 0))
            adj = F.relu(F.leaky_relu(a))

            adj = adj - torch.diag(torch.diag(adj))

            # Essential Correlation Information Focusing
            values, indices = torch.topk(adj.reshape(-1), int(num_elements * self.w_ratio), largest=True)
            mask = torch.zeros_like(adj.reshape(-1), device=adj.device)
            mask[indices] = 1
            adj = mask.view(adj.size(0), adj.size(1)).view_as(adj) * adj

            adj = adj + torch.eye(adj.size(0)).to(adj.device)

            d = adj.sum(1)
            adj = adj / d.view(-1, 1)

            adjs.append(adj)

        time_indices = time_indices % self.num_adj_matrices
        oadj = torch.stack(adjs)
        dadj = oadj[time_indices]
        return dadj


class MTT(nn.Module):
    def __init__(self, patch_num, patch_len, n_layers=3, d_model=128, n_heads=16, d_k=None, d_v=None,
                 d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., act="gelu", store_attn=False,
                 res_attention=True, pre_norm=False,
                 pe='zeros', learn_pe=True, **kwargs):
        super().__init__()

        self.patch_num = patch_num
        self.patch_len = patch_len

        # Input encoding
        q_len = patch_num
        self.W_P = nn.Linear(patch_len, d_model)
        self.seq_len = q_len

        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, q_len, d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder1 = TransformerEncoder(d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                           attn_dropout=attn_dropout, dropout=dropout,
                                           pre_norm=pre_norm, activation=act, res_attention=res_attention,
                                           n_layers=n_layers,
                                           store_attn=store_attn)
        self.encoder2 = TransformerEncoder(d_model * 2, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                           attn_dropout=attn_dropout, dropout=dropout,
                                           pre_norm=pre_norm, activation=act, res_attention=res_attention,
                                           n_layers=n_layers,
                                           store_attn=store_attn)
        self.encoder3 = TransformerEncoder(d_model * 4, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                           attn_dropout=attn_dropout, dropout=dropout,
                                           pre_norm=pre_norm, activation=act, res_attention=res_attention,
                                           n_layers=n_layers,
                                           store_attn=store_attn)

    def forward(self, x) -> Tensor:  # x: [bs x nvars x patch_len x patch_num]
        n_vars = x.shape[1]
        # Input encoding
        x = x.permute(0, 1, 3, 2)  # x: [bs x nvars x patch_num x patch_len]
        x = self.W_P(x)  # x: [bs x nvars x patch_num x d_model]
        # print("mttx",x.shape)

        u = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))  # u: [bs * nvars x patch_num x d_model]

        u = self.dropout(u + self.W_pos)  # u: [bs * nvars x patch_num x d_model]

        # Encoder
        u = self.encoder1(u)  # u: [bs * nvars x patch_num x d_model]

        # Patch Combination
        b, n, d = u.shape
        u = u.reshape(b, n // 2, 2 * d)

        # Encoder
        u = self.encoder2(u)  # u: [bs * nvars x patch_num / 2 x d_model * 2]

        # Patch Combination
        b, n, d = u.shape
        u = u.reshape(b, n // 2, 2 * d)

        # Encoder
        u = self.encoder3(u)  # u: [bs * nvars x patch_num / 4 x d_model * 4]

        u = torch.reshape(u, (-1, n_vars, u.shape[-2], u.shape[-1]))  # u: [bs x nvars x patch_num / 4 x d_model * 4]
        u = u.permute(0, 1, 3, 2)  # u: [bs x nvars x d_model * 4 x patch_num / 4]

        return u

    # Cell


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, d_ff=None,
                 norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                 res_attention=False, n_layers=1, pre_norm=False, store_attn=False):
        super().__init__()

        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                     attn_dropout=attn_dropout, dropout=dropout,
                                     activation=activation, res_attention=res_attention,
                                     pre_norm=pre_norm, store_attn=store_attn) for i in range(n_layers)])
        self.res_attention = res_attention

    def forward(self, src: Tensor, key_padding_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None):
        output = src
        scores = None
        if self.res_attention:
            for mod in self.layers: output, scores = mod(output, prev=scores, key_padding_mask=key_padding_mask,
                                                         attn_mask=attn_mask)
            return output
        else:
            for mod in self.layers: output = mod(output, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            return output


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, d_ff=256, store_attn=False,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, activation="gelu", res_attention=False,
                 pre_norm=False):
        super().__init__()
        assert not d_model % n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        # Multi-Head attention
        self.res_attention = res_attention
        self.self_attn = _MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout,
                                             proj_dropout=dropout, res_attention=res_attention)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn

    def forward(self, src: Tensor, prev: Optional[Tensor] = None, key_padding_mask: Optional[Tensor] = None,
                attn_mask: Optional[Tensor] = None) -> Tensor:

        # Multi-Head attention sublayer
        if self.pre_norm:
            src = self.norm_attn(src)
        # Multi-Head attention
        if self.res_attention:
            src2, attn, scores = self.self_attn(src, src, src, prev, key_padding_mask=key_padding_mask,
                                                attn_mask=attn_mask)
        else:
            src2, attn = self.self_attn(src, src, src, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        if self.store_attn:
            self.attn = attn
        # Add & Norm
        src = src + self.dropout_attn(src2)  # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_attn(src)

        # Feed-forward sublayer
        if self.pre_norm:
            src = self.norm_ffn(src)
        # Position-wise Feed-Forward
        src2 = self.ff(src)
        # Add & Norm
        src = src + self.dropout_ffn(src2)  # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_ffn(src)

        if self.res_attention:
            return src, scores
        else:
            return src


class _MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, res_attention=False, attn_dropout=0., proj_dropout=0.,
                 qkv_bias=True, lsa=False):
        """Multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs) x max_q_len x d_model]
            K, V:    [batch_size (bs) x q_len x d_model]
            mask:    [q_len x q_len]
        """
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        # Scaled Dot-Product Attention (multiple heads)
        self.res_attention = res_attention
        self.sdp_attn = _ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout,
                                                   res_attention=self.res_attention, lsa=lsa)

        # Poject output
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))

    def forward(self, Q: Tensor, K: Optional[Tensor] = None, V: Optional[Tensor] = None, prev: Optional[Tensor] = None,
                key_padding_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None):

        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q

        # Linear (+ split in multiple heads)
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,
                                                                         2)  # q_s    : [bs x n_heads x max_q_len x d_k]
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0, 2, 3,
                                                                       1)  # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1, 2)  # v_s    : [bs x n_heads x q_len x d_v]

        # Apply Scaled Dot-Product Attention (multiple heads)
        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev,
                                                              key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            output, attn_weights = self.sdp_attn(q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        # output: [bs x n_heads x q_len x d_v], attn: [bs x n_heads x q_len x q_len], scores: [bs x n_heads x max_q_len x q_len]

        # back to the original inputs dimensions
        output = output.transpose(1, 2).contiguous().view(bs, -1,
                                                          self.n_heads * self.d_v)  # output: [bs x q_len x n_heads * d_v]
        output = self.to_out(output)

        if self.res_attention:
            return output, attn_weights, attn_scores
        else:
            return output, attn_weights


class _ScaledDotProductAttention(nn.Module):
    r"""Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual attention from previous layer
    (Realformer: Transformer likes residual attention by He et al, 2020) and locality self sttention (Vision Transformer for Small-Size Datasets
    by Lee et al, 2021)"""

    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa

    def forward(self, q: Tensor, k: Tensor, v: Tensor, prev: Optional[Tensor] = None,
                key_padding_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None):
        '''
        Input shape:
            q               : [bs x n_heads x max_q_len x d_k]
            k               : [bs x n_heads x d_k x seq_len]
            v               : [bs x n_heads x seq_len x d_v]
            prev            : [bs x n_heads x q_len x seq_len]
            key_padding_mask: [bs x seq_len]
            attn_mask       : [1 x seq_len x seq_len]
        Output shape:
            output:  [bs x n_heads x q_len x d_v]
            attn   : [bs x n_heads x q_len x seq_len]
            scores : [bs x n_heads x q_len x seq_len]
        '''

        # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        attn_scores = torch.matmul(q, k) * self.scale  # attn_scores : [bs x n_heads x max_q_len x q_len]

        # Add pre-softmax attention scores from the previous layer (optional)
        if prev is not None: attn_scores = attn_scores + prev

        # Attention mask (optional)
        if attn_mask is not None:  # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        # Key padding mask (optional)
        if key_padding_mask is not None:  # mask with shape [bs x q_len] (only when max_w_len == q_len)
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)

        # normalize the attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)  # attn_weights   : [bs x n_heads x max_q_len x q_len]
        attn_weights = self.attn_dropout(attn_weights)

        # compute the new values given the attention weights
        output = torch.matmul(attn_weights, v)  # output: [bs x n_heads x max_q_len x d_v]

        if self.res_attention:
            return output, attn_weights, attn_scores
        else:
            return output, attn_weights


class Predictor(nn.Module):
    def __init__(self, n_vars, nf, target_window, predictor_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(predictor_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]

        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)

        return x
