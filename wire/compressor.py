"""Selective scan compressor (paper Sec. 3.2).

For every frame t with eta_t visual tokens the compressor emits k_t refined
tokens:

  1. Query init: bilinear resampling of the frame's token grid to a_t x b_t
     (a_t * b_t = k_t, aspect ratio closest to the grid's). No parameters.
  2. The queries are interleaved uniformly among the same frame's full
     resolution tokens; the question embeddings are placed at both ends of
     the sequence; one bidirectional Mamba2 layer refines the whole sequence.
  3. The refined queries Q_t are read back at the query positions; the
     refined full resolution rows F_t are returned for the cross attention
     readout of the hypernetworks.

The construction follows BIMBA (Islam et al., 2025) with two changes forced
by Qwen3-VL's variable frame grids: interleaving is done per frame rather
than globally, and the question is part of the scanned sequence.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


# ----------------------------------------------------------------------------
# frame segmentation and query initialization
# ----------------------------------------------------------------------------
def video_frame_runs(vpos: torch.Tensor, tokens_per_frame: int):
    """Split a sample's visual positions into frames.

    Qwen3-VL separates frames with timestamp text, so each frame is a
    contiguous run of visual positions; run widths vary slightly with the
    aspect ratio. If the whole video is one run (no separators), fall back to
    fixed chunks of `tokens_per_frame`.

    Returns run_id [n] (non-decreasing) and run_len [R].
    """
    n = int(vpos.numel())
    dev = vpos.device
    gaps = (vpos[1:] - vpos[:-1]) > 1
    run_id = torch.cat([torch.zeros(1, dtype=torch.long, device=dev),
                        gaps.cumsum(0)])
    R = int(run_id[-1]) + 1
    if R == 1 and n > tokens_per_frame:
        run_id = torch.arange(n, device=dev) // tokens_per_frame
        R = int(run_id[-1]) + 1
    run_len = torch.bincount(run_id, minlength=R)
    return run_id, run_len


def frame_budgets(run_len: torch.Tensor, k: int, tokens_per_frame: int):
    """k_t = min(eta_t, max(1, round(eta_t * k / eta)))  (Eq. 3)."""
    kt = torch.clamp((run_len.float() * k / tokens_per_frame).round().long(),
                     min=1)
    return torch.minimum(kt, run_len)


def contraction_segments(run_id: torch.Tensor, run_len: torch.Tensor,
                         kt: torch.Tensor):
    """Segment map of the sequence contraction (Sec. 3.3, Eq. 6).

    Frame run r is split into kt[r] consecutive near-equal segments: the
    j-th position of the run (from 0) lies in segment floor(j * kt[r] /
    run_len[r]). Each segment's refined token takes the segment's last
    position.

    Returns seg [n] (segment of every visual position, numbered over all
    runs) and last [m] (index into the visual positions of each segment's
    last position), m = kt.sum().
    """
    dev = run_id.device
    run_start = torch.cumsum(run_len, 0) - run_len
    koff = torch.cumsum(kt, 0) - kt
    jin = torch.arange(run_id.numel(), device=dev) - run_start[run_id]
    seg = koff[run_id] + (jin * kt[run_id]) // run_len[run_id]
    m = int(kt.sum())
    last = torch.searchsorted(seg, torch.arange(1, m + 1, device=dev)) - 1
    return seg, last


def _aspect_factor(k: int, gh: int, gw: int) -> Tuple[int, int]:
    """Factor k = a * b with a / b closest to gh / gw."""
    best, err = (1, k), float("inf")
    for a in range(1, int(math.isqrt(k)) + 1):
        if k % a:
            continue
        for cand in ((a, k // a), (k // a, a)):
            e = abs(math.log((cand[0] / cand[1]) / (gh / gw)))
            if e < err:
                best, err = cand, e
    return best


def bilinear_queries(ve: torch.Tensor, run_start: torch.Tensor,
                     run_len: torch.Tensor, kt: torch.Tensor,
                     grid: Optional[Tuple[int, int]]) -> torch.Tensor:
    """Initial queries Q_t^(0): bilinear resampling of each frame's grid.

    `grid` is the post-merge (h, w) token grid of a frame. Frames whose run
    width does not match the grid fall back to 1-D linear resampling of the
    flattened run. Returns [sum(kt), D] in float32.
    """
    D = ve.shape[-1]
    gh, gw = grid if grid is not None else (0, 0)
    R = int(run_len.numel())
    # Batched path: all frames share one width and one budget (the usual case).
    if R > 1 and bool((run_len == run_len[0]).all()) \
            and bool((kt == kt[0]).all()):
        L, k = int(run_len[0]), int(kt[0])
        if ve.shape[0] >= R * L:
            x = ve[:R * L]
            if gh and gh * gw == L and k > 1:
                a, b = _aspect_factor(k, gh, gw)
                y = F.interpolate(
                    x.view(R, gh, gw, D).permute(0, 3, 1, 2).float(),
                    size=(a, b), mode="bilinear", align_corners=False)
                return y.permute(0, 2, 3, 1).reshape(R * k, D)
            y = F.interpolate(x.view(R, L, D).transpose(1, 2).float(),
                              size=k, mode="linear", align_corners=False)
            return y.transpose(1, 2).reshape(R * k, D)
    outs = []
    for r in range(R):
        s, L, k = int(run_start[r]), int(run_len[r]), int(kt[r])
        x = ve[s:s + L]
        if gh and gh * gw == L and k > 1:
            a, b = _aspect_factor(k, gh, gw)
            y = F.interpolate(
                x.view(gh, gw, D).permute(2, 0, 1).unsqueeze(0).float(),
                size=(a, b), mode="bilinear", align_corners=False)
            outs.append(y.squeeze(0).permute(1, 2, 0).reshape(k, D))
        else:
            y = F.interpolate(x.t().unsqueeze(0).float(), size=k,
                              mode="linear", align_corners=False)
            outs.append(y.squeeze(0).t())
    return torch.cat(outs, 0)


# ----------------------------------------------------------------------------
# the scan
# ----------------------------------------------------------------------------
def _interleave_mask(n: int, m: int, device) -> torch.Tensor:
    """Positions of m queries spread uniformly over n + m slots."""
    total = n + m
    mask = torch.zeros(total, dtype=torch.bool, device=device)
    if m == 0:
        return mask
    slots = torch.linspace(0, total - 1, m + 1, device=device).long()[1:]
    mask[slots] = True
    if int(mask.sum()) != m:
        # linspace collides once the spacing drops below one slot
        mask.zero_()
        mask[-m:] = True
    return mask


class SelectiveScanCompressor(nn.Module):
    """One bidirectional Mamba2 layer over [E_q, iota(G_1, Q_1), ..., E_q].

    Y = Z + SSM_fwd(LN(Z)) + flip(SSM_bwd(flip(LN(Z))))      (Eq. 5)
    """

    def __init__(self, hidden_size: int, d_state: int = 64, d_conv: int = 4,
                 expand: int = 2, headdim: int = 128, n_layer: int = 1,
                 out_proj_init_std: float = 0.0):
        super().__init__()
        if n_layer != 1:
            raise ValueError("only the single-layer compressor is supported")
        from mamba_ssm import Mamba2

        def mixer():
            return Mamba2(d_model=hidden_size, d_state=d_state, d_conv=d_conv,
                          expand=expand, headdim=headdim)

        # ModuleLists keep the parameter names of the released checkpoints
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_size)])
        self.fwd = nn.ModuleList([mixer()])
        self.bwd = nn.ModuleList([mixer()])
        for blk in (self.fwd[0], self.bwd[0]):
            if out_proj_init_std > 0:
                nn.init.normal_(blk.out_proj.weight, std=out_proj_init_std)
            else:
                nn.init.zeros_(blk.out_proj.weight)

    def forward(self, video: torch.Tensor, query: torch.Tensor,
                run_id: torch.Tensor, q_run_id: torch.Tensor,
                text: Optional[torch.Tensor] = None,
                bidirectional: bool = True):
        """video [n, D] full resolution tokens, query [m, D] initial queries,
        run_id [n] / q_run_id [m] frame index of each (both non-decreasing),
        text [t, D] question embeddings or None.

        Returns (refined queries [m, D], refined full resolution rows [n, D],
        frame index of each row [n]).
        """
        out_dtype = query.dtype
        wdtype = self.norms[0].weight.dtype
        v, q = video.to(wdtype), query.to(wdtype)
        t = text.to(wdtype) if text is not None and text.numel() else None

        R = int(run_id.max()) + 1
        cv = torch.bincount(run_id, minlength=R)
        cq = torch.bincount(q_run_id, minlength=R)

        # per-frame interleave: frame r occupies a block of cv[r] + cq[r] slots
        qmask = None
        if bool((cv == cv[0]).all() and (cq == cq[0]).all()) and int(cq[0]) > 0:
            mk0 = _interleave_mask(int(cv[0]), int(cq[0]), q.device)
            if int(mk0.sum()) == int(cq[0]):
                qmask = mk0.repeat(R)
        if qmask is not None and int(qmask.sum()) == q.shape[0]:
            comb = torch.empty(qmask.numel(), v.shape[-1], device=q.device,
                               dtype=v.dtype)
            comb[qmask] = q
            comb[~qmask] = v
        else:                               # frames of different widths
            parts, mparts, vo, qo = [], [], 0, 0
            for r in range(R):
                nv, nq = int(cv[r]), int(cq[r])
                if nv == 0 and nq == 0:
                    continue
                mk = _interleave_mask(nv, nq, q.device)
                blk = torch.empty(nv + nq, v.shape[-1], device=q.device,
                                  dtype=v.dtype)
                blk[mk] = q[qo:qo + nq]
                blk[~mk] = v[vo:vo + nv]
                parts.append(blk)
                mparts.append(mk)
                vo += nv
                qo += nq
            comb = torch.cat(parts, 0)
            qmask = torch.cat(mparts, 0)

        if t is not None:                   # question at both ends
            pad = torch.zeros(t.shape[0], dtype=torch.bool, device=q.device)
            comb = torch.cat([t, comb, t], 0)
            qmask = torch.cat([pad, qmask, pad], 0)

        h = self.norms[0](comb).unsqueeze(0)
        if bidirectional:
            h = self.fwd[0](h) + self.bwd[0](h.flip(1)).flip(1)
        else:
            h = self.fwd[0](h)
        comb = comb + h.squeeze(0)

        vmask = ~qmask
        if t is not None:
            vmask[: t.shape[0]] = False
            vmask[-t.shape[0]:] = False
        return (comb[qmask].to(out_dtype), comb[vmask].to(out_dtype),
                run_id)
