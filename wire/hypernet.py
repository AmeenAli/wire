"""Hypernetworks and the cross attention readout (paper Sec. 3.4)."""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn


class LayerHyperNet(nn.Module):
    """Generates rank-r updates for the seven linear maps of one layer.

    h -> z = SiLU(W2 SiLU(W1 LN(h)))                     trunk, z in R^512
    Z_m = rownorm(reshape_{r x d_z}(W_m^lat z))          rank latents
    A_m = Z_m U_m,  B_m = Z_m V_m                        (Eq. 7)
    Delta W_m = (alpha / r) B_m^T A_m

    V_m (proj_B) is zero-initialized, so Delta W_m = 0 at step zero.
    """

    def __init__(self, cond_size: int, dims: Dict[str, Tuple[int, int]],
                 rank: int = 64, trunk_width: int = 512,
                 latent_size: int = 256):
        super().__init__()
        self.rank = rank
        self.latent_size = latent_size
        self.dims = dims                                   # {name: (d_in, d_out)}
        self.trunk = nn.Sequential(
            nn.LayerNorm(cond_size),
            nn.Linear(cond_size, trunk_width),
            nn.SiLU(),
            nn.Linear(trunk_width, trunk_width),
            nn.SiLU(),
        )
        self.rank_head = nn.ModuleDict({
            m: nn.Linear(trunk_width, rank * latent_size) for m in dims})
        self.proj_A = nn.ModuleDict()                      # U_m
        self.proj_B = nn.ModuleDict()                      # V_m
        for m, (d_in, d_out) in dims.items():
            self.proj_A[m] = nn.Linear(latent_size, d_in, bias=False)
            self.proj_B[m] = nn.Linear(latent_size, d_out, bias=False)
            nn.init.normal_(self.proj_A[m].weight, std=1.0 / math.sqrt(latent_size))
            nn.init.zeros_(self.proj_B[m].weight)

    def forward(self, h: torch.Tensor):
        """h [B, cond_size] -> {name: (A [B, r, d_in], B [B, r, d_out])}."""
        z = self.trunk(h.to(self.trunk[1].weight.dtype))
        out = {}
        for m in self.dims:
            lat = self.rank_head[m](z).view(-1, self.rank, self.latent_size)
            lat = F.normalize(lat, dim=-1)
            out[m] = (self.proj_A[m](lat), self.proj_B[m](lat))
        return out


class FrameCrossPool(nn.Module):
    """h_c: contracted positions attend into their own frame's refined full
    resolution rows F_t, then are averaged (Eq. 6).

    The attention is block diagonal by frame. Keys and values depend only on
    the compressor output, so `prepare` computes them once per forward pass
    and all hypernetwork layers share them.
    """

    def __init__(self, hidden_size: int, dim: int = 1024, heads: int = 8,
                 out_proj_init_std: float = 0.0):
        super().__init__()
        self.heads, self.dim = heads, dim
        self.q_proj = nn.Linear(hidden_size, dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, dim, bias=False)
        self.o_proj = nn.Linear(dim, hidden_size, bias=False)
        self.norm = nn.LayerNorm(hidden_size)
        if out_proj_init_std > 0:
            nn.init.normal_(self.o_proj.weight, std=out_proj_init_std)
        else:
            nn.init.zeros_(self.o_proj.weight)

    def prepare(self, kv_tok: torch.Tensor, kv_run: torch.Tensor,
                q_run: torch.Tensor):
        """Project the full resolution rows once. Returns None when frames
        differ in width (the caller then uses the plain mean)."""
        R = int(kv_run.max()) + 1 if kv_run.numel() else 0
        if R == 0:
            return None
        cq = torch.bincount(q_run, minlength=R)
        ck = torch.bincount(kv_run, minlength=R)
        if not bool((cq == cq[0]).all() and (ck == ck[0]).all()):
            return None
        nq, nk = int(cq[0]), int(ck[0])
        if nq == 0 or nk == 0:
            return None
        dt = self.k_proj.weight.dtype
        D = kv_tok.shape[-1]
        ks = kv_tok[torch.argsort(kv_run, stable=True)].view(R, nk, D).to(dt)
        kh = self.k_proj(ks).view(R, nk, self.heads, -1).transpose(1, 2)
        vh = self.v_proj(ks).view(R, nk, self.heads, -1).transpose(1, 2)
        return kh, vh, torch.argsort(q_run, stable=True), R, nq

    def forward(self, q_tok: torch.Tensor, cache) -> torch.Tensor:
        """q_tok [m, D] queries for one sample -> pooled vector [D]."""
        if cache is None:
            return q_tok.mean(0)
        kh, vh, q_order, R, nq = cache
        D = q_tok.shape[-1]
        dt = self.q_proj.weight.dtype
        qs = q_tok[q_order].view(R, nq, D)
        qh = self.q_proj(self.norm(qs).to(dt)) \
            .view(R, nq, self.heads, -1).transpose(1, 2)
        a = F.scaled_dot_product_attention(qh, kh, vh)
        a = a.transpose(1, 2).reshape(R, nq, self.dim)
        enriched = qs + self.o_proj(a).to(qs.dtype)
        return enriched.reshape(-1, D).mean(0)
