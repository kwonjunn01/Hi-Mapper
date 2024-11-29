import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from timm.models.layers import drop, drop_path, trunc_normal_

   
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)


        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v):
        B, N, C = q.shape
        assert k.shape == v.shape
        B, M, C = k.shape
        q = self.q_proj(q).reshape(B, N, self.num_heads, C // self.num_heads)
        k = self.k_proj(k).reshape(B, M, self.num_heads, C // self.num_heads)
        v = self.v_proj(v).reshape(B, M, self.num_heads, C // self.num_heads)

        attn = torch.einsum('bnkc,bmkc->bknm', q, k) * self.scale

        attn = attn.softmax(dim=-1)

        x = torch.einsum('bknm,bmkc->bnkc', attn, v).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dropout=0.1,
    ):
        super().__init__()
        self.self_attn = Attention(d_model, nhead, proj_drop=dropout)
        self.cross_attn = Attention(d_model, nhead, proj_drop=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.memnorm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x, mem):
        q = self.norm1(x)
        x = x + self.cross_attn(q, mem, mem)
        q = k = v = self.norm2(x)
        x = x + self.self_attn(q, k, v)
        x = x + self.dropout(self.mlp(self.norm3(x)))
        return x
    
class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dropout=0.1,
    ):
        super().__init__()
        self.self_attn = Attention(d_model, nhead, proj_drop=dropout)
        self.cross_attn = Attention(d_model, nhead,qk_scale=1/0.07, proj_drop=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x, mem):
        q = self.norm1(x)
        x = x + self.cross_attn(q, mem, mem)
        q = k = v = self.norm2(x)
        x = x + self.self_attn(q, k, v)
        x = x + self.dropout(self.mlp(self.norm3(x)))
        return x
class XTransformerDecoder(nn.Module):
    def __init__(self,
                 transformer_width=128,
                 transformer_heads=4,
                 transformer_layers=4,
                 visual_dim=384,
                 dropout=0.1,
                 **kwargs):
        super().__init__()

        self.memory_proj = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, transformer_width),
            nn.LayerNorm(transformer_width),
        )

        self.node_proj = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, transformer_width),
        )

        self.decoder = nn.ModuleList([
                    TransformerDecoderLayer(transformer_width, transformer_heads, dropout) for _ in range(1)
                ])
        
        self.out_proj = nn.Sequential(
            nn.LayerNorm(transformer_width),
            nn.Linear(transformer_width, visual_dim)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    def kl_divergence(self, mu, logsigma):
        return -0.5 * (1 + logsigma - mu.pow(2) - logsigma.exp()).mean()
    def reg_logsigma(self, logsigma):
        return logsigma.mean()
    def forward(self, node_mean, node_std, kv):
        B, N, C = kv.shape
        kv = self.memory_proj(kv)
        nodes = (node_mean.expand(B,-1,-1) + torch.exp(node_std*.5).expand(B, -1, -1)*torch.randn_like(node_std, device=node_std.device, dtype=node_std.dtype))
        q = self.node_proj(nodes)
        node = []
        node_n = []
        prev_mean = 0
        prev_std = 0
        for layer in self.decoder:
            for idx in range(4):
                tnodes = torch.zeros_like(nodes, device=nodes.device, dtype= nodes.dtype)
                if idx == 0:
                    q = layer(q, kv)
                    prev_mean = node_mean
                    prev_std = node_std
                else:
                    q_tmp = 0
                    curr_mean = 0
                    curr_std = 0
                    term = 2**(idx-1)
                    sterm = 2**idx
                    

                    for i in range(term):
                        q_tmp += q[:,i::term]
                    q_tmp = q_tmp/term
                    node.append(self.out_proj(q_tmp))
                    node_n.append(self.out_proj(q))
                    ## define l-level seed nodes
                    for i in range(2):
                        curr_mean += prev_mean[i::2]
                        curr_std += prev_std[i::2].exp() + prev_mean[i::2]**2
                    curr_mean = curr_mean/2
                    curr_std = torch.log(curr_std/2 - curr_mean**2)
                    for i in range(sterm):
                        tnodes[:,i::sterm] = (curr_mean.expand(B,-1,-1) + torch.exp(curr_std*.5).expand(B, -1, -1)*torch.randn_like(curr_std, device=node_std.device, dtype=node_std.dtype))
                    
                        
                        
                    prev_mean = curr_mean
                    prev_std = curr_std
                    q = self.node_proj(tnodes)
                    # q = torch.repeat_interleave(q, term, dim=1)
                    q = layer(q, kv)
                    # q_tmp = (q[0::2] + q[1::2])/2
        kl = self.kl_divergence(curr_mean, curr_std)
        term = 2**(idx)
        q_tmp = 0
        for i in range(term):
            q_tmp += q[:,i::term]
        q_tmp = q_tmp/term
        node.append(self.out_proj(q_tmp))
        node_n.append(self.out_proj(q))
        node = torch.cat(node, dim = 1)    
        return [node, node_n],kl
class XTransformerEncoder(nn.Module):
    def __init__(self,
                 transformer_width=128,
                 transformer_heads=4,
                 transformer_layers=4,
                 visual_dim=384,
                 dropout=0.1,
                 **kwargs):
        super().__init__()

        self.memory_proj = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, transformer_width),
            nn.LayerNorm(transformer_width),
        )

        self.node_proj = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, transformer_width),
        )

        self.decoder = nn.ModuleList([
                    TransformerEncoderLayer(transformer_width, transformer_heads, dropout) for _ in range(transformer_layers)
                ])
        
        self.out_proj = nn.Sequential(
            nn.LayerNorm(transformer_width),
            nn.Linear(transformer_width, visual_dim)
        )
        self.dropout = nn.Dropout(p=0.2)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    
    def forward(self, query, node):
        # B, N, C = kv.shape
        out = []
        q = self.node_proj(query)
        # ori = q
        # out.append(self.out_proj(q))
        for idx, layer in enumerate(self.decoder):
            kv = node[idx]
            torch.repeat_interleave(kv, 2**idx, dim=1)
            kv = self.memory_proj(kv)
            q = layer(q, kv)
            out.append(self.out_proj(q))
            # q = ori + q
        out = torch.stack(out, dim = 1)
        return out
