"""
Attention as RNN (https://arxiv.org/abs/2405.13956) 
using _higher_order_ops.associative_scan
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch._higher_order_ops.associative_scan import associative_scan
from associative_scan import associative_scan
from einops import rearrange, repeat

def associate(a, b):
    ma0, ua0, wa0 = a[..., :1], a[..., 1:2], a[..., 2:]
    mb0, ub0, wb0 = b[..., :1], b[..., 1:2], b[..., 2:]

    m = torch.max(ma0, mb0).detach()  
    u = union(ua0, ma0, ub0, mb0)  # denominator
    w = union(wa0, ma0, wb0, mb0)  # numerator
    return torch.cat([m, u, w], dim=-1)

def union(ua, ma, ub, mb):
    m_aub = torch.max(ma, mb).detach()
    ua_exp = ua * torch.exp(ma - m_aub)
    ub_exp = ub * torch.exp(mb - m_aub)
    return ua_exp + ub_exp

class attention_as_rnn(nn.Module):

    def __init__(
        self,
        heads,
        dim,
        dim_inner,
        dim_out,
        activation="none",
        **kwargs,
    ):
        super().__init__()

        self.heads = heads
        self.dim = dim
        self.dim_inner = dim_inner
        self.dim_out = dim_out
        self.activation = getattr(F, activation) if activation != "none" else lambda x: x

        self.kv_kernel = nn.Parameter(torch.Tensor(dim, self.heads, self.dim_inner, 2))
        self.q_kernel = nn.Parameter(torch.Tensor(self.heads, self.dim_inner))
        self.to_out = nn.Linear(self.dim_inner, self.dim_out)

        nn.init.xavier_normal_(self.kv_kernel)
        nn.init.xavier_normal_(self.q_kernel)

    def forward(self, inputs):
        shape = inputs.size()
        B, T = shape[0], shape[1]

        q = self.q_kernel
        kv = torch.einsum("bti,ihok->bthok", inputs, self.kv_kernel)
        kv = self.activation(kv)

        k, v = kv.split(1, dim=-1)
        k, v = k[..., 0], v[..., 0]

        st = torch.einsum("hd,bthd->bth", q, k).unsqueeze(-1)
        u_init = torch.ones(B, T, self.heads, 1).to(inputs.device)
        i = torch.cat([st, u_init, v], dim=-1)

        o = associative_scan(associate, i, dim=1, combine_mode="generic")
        m, c, a = o[..., :1], o[..., 1:2], o[..., 2:]
        h = a / c

        h = h.mean(dim=-2) 
        h = self.to_out(h)

        return h

if __name__ == "__main__":
  scan_attention = attention_as_rnn(heads=4, dim=14, dim_inner=256, dim_out=32).cuda()
  data = torch.rand(32, 1000, 14).cuda()
  print(scan_attention(data).shape)
