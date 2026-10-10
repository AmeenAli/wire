#!/usr/bin/env python
"""CPU check: numbers of the paper, rebuilt with the package's own functions.

Needs only PyTorch (no GPU, no download) and runs in a few seconds:

    python reproduce/cpu_check.py

  1. Sequence contraction at 256 frames (Sections 3.1 and 3.3, Appendix D):
     the paper's decoder input of 15,394 positions, 14,336 of them visual,
     contracts to 5,154 positions with 16 tokens per frame and to 2,594 with 6.
  2. Adapted layers (Section 3.4): one hypernetwork at each of the layers
     3, 7, ..., 35 of Qwen3-VL-8B, seven maps each, 63 maps in all.
  3. Trained parameters (Section 4): 763M in the nine hypernetworks and
     16.8M in the cross-attention readout, built on PyTorch's meta device with
     the dimensions of Qwen3-VL-8B. The compressor (203M, 0.98B in all) is
     counted when the Mamba2 kernels (mamba-ssm) are installed.
  4. Updates (Sections 3.4 and 3.5): the generated update is exactly zero at
     initialization, and after the merge the plain maps reproduce the adapted
     maps of prefill.

Exits with status 1 if a number differs from the paper.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch
from torch import nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from wire.compressor import (contraction_segments, frame_budgets,  # noqa: E402
                             video_frame_runs)
from wire.config import WireConfig  # noqa: E402
from wire.hypernet import FrameCrossPool, LayerHyperNet  # noqa: E402
from wire.modeling import (ATTN_MODULES, MLP_MODULES, WireState,  # noqa: E402
                           _lora_linear, merge_updates, unmerge_updates)

# Qwen3-VL-8B-Instruct: 36 decoder layers, hidden size 4096, 32 query and 8
# key-value heads of width 128 (Section 3.1), and MLP width 12,288 (the
# model's configuration).
BACKBONE = dict(layers=36, hidden=4096, heads=32, kv_heads=8, head_dim=128,
                intermediate=12288)
PAPER = {"positions": 15394, "visual": 14336, "contracted": {16: 5154, 6: 2594},
         "layers": list(range(3, 36, 4)), "maps": 63,
         "hypernetworks": 763e6, "cross_pool": 16.8e6, "compressor": 203e6,
         "total": 0.98e9}


# --------------------------------------------------------------------------- #
# 1. sequence contraction
# --------------------------------------------------------------------------- #
def contraction_lengths(k: int, frames: int = 256, eta: int = 56,
                        positions: int = PAPER["positions"]) -> dict:
    """Decoder input length before and after contraction.

    The input has `frames` runs of `eta` visual positions (the paper's
    nominal eta = 56) and `positions - frames * eta` text positions; one text
    position separates consecutive runs. The kept positions are every text
    position plus the last position of each segment (wire.compressor).
    """
    visual = frames * eta
    text = positions - visual
    lead = text - (frames - 1)
    vpos = torch.cat([lead + t * (eta + 1) + torch.arange(eta)
                      for t in range(frames)])
    run_id, run_len = video_frame_runs(vpos, eta)
    kt = frame_budgets(run_len, k, eta)
    seg, last = contraction_segments(run_id, run_len, kt)
    return {"positions": positions, "visual": visual, "runs": int(run_len.numel()),
            "kept_per_frame": int(kt[0]), "contracted": text + int(last.numel())}


# --------------------------------------------------------------------------- #
# 2. and 3. adapted layers and trained parameters
# --------------------------------------------------------------------------- #
def layer_dims(b: dict = BACKBONE) -> dict:
    """(d_in, d_out) of the seven adapted maps of one decoder layer."""
    d, q, kv, ff = (b["hidden"], b["heads"] * b["head_dim"],
                    b["kv_heads"] * b["head_dim"], b["intermediate"])
    return {"q_proj": (d, q), "k_proj": (d, kv), "v_proj": (d, kv),
            "o_proj": (q, d), "gate_proj": (d, ff), "up_proj": (d, ff),
            "down_proj": (ff, d)}


def trained_parameters(cfg: WireConfig = None, b: dict = BACKBONE) -> dict:
    """Parameter counts of the trained modules; the compressor is None when
    the Mamba2 kernels (mamba-ssm) are not installed."""
    cfg = cfg or WireConfig()
    layers = cfg.target_layers(b["layers"])
    count = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
    with torch.device("meta"):
        hyper = sum(count(LayerHyperNet(3 * b["hidden"], layer_dims(b), rank=cfg.rank,
                                        trunk_width=cfg.trunk_width,
                                        latent_size=cfg.latent_size))
                    for _ in layers)
        pool = count(FrameCrossPool(b["hidden"], dim=cfg.cross_pool_dim,
                                    heads=cfg.cross_pool_heads))
        try:
            from wire.compressor import SelectiveScanCompressor
            scan = count(SelectiveScanCompressor(
                b["hidden"], d_state=cfg.scan_d_state, d_conv=cfg.scan_d_conv,
                expand=cfg.scan_expand, headdim=cfg.scan_headdim))
        except Exception:                      # no Mamba2 kernels on this machine
            scan = None
    return {"layers": layers, "maps": len(layers) * len(layer_dims(b)),
            "hypernetworks": hyper, "cross_pool": pool, "compressor": scan}


# --------------------------------------------------------------------------- #
# 4. the update at step zero and the merge after prefill
# --------------------------------------------------------------------------- #
class _Layer(nn.Module):
    """A decoder layer reduced to the seven maps WIRE adapts, with the names
    of Qwen3-VL (self_attn.q_proj, ..., mlp.down_proj)."""

    def __init__(self, dims: dict):
        super().__init__()
        self.self_attn, self.mlp = nn.Module(), nn.Module()
        for name, (d_in, d_out) in dims.items():
            parent = self.self_attn if name in ATTN_MODULES else self.mlp
            setattr(parent, name, nn.Linear(d_in, d_out, bias=False))

    def map(self, name: str) -> nn.Linear:
        return getattr(self.self_attn if name in ATTN_MODULES else self.mlp, name)


def merge_check(seed: int = 0, hidden: int = 64, intermediate: int = 96,
                rank: int = 8, n_layers: int = 8, tokens: int = 5) -> dict:
    """Generate updates with the package's hypernetworks for a small decoder,
    then compare the adapted maps of prefill (Eq. 10) with the plain maps
    after `merge_updates`, and the weights after `unmerge_updates` with the
    originals."""
    torch.manual_seed(seed)
    dims = layer_dims(dict(hidden=hidden, heads=4, kv_heads=2, head_dim=hidden // 4,
                           intermediate=intermediate))
    cfg = WireConfig(rank=rank, alpha=2 * rank, layer_stride=4)
    model = nn.Module()
    model.model = nn.Module()
    model.model.language_model = nn.Module()
    model.model.language_model.layers = nn.ModuleList(_Layer(dims) for _ in range(n_layers))
    model.wire_state = st = WireState(cfg)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}

    h = torch.randn(1, 3 * hidden)             # the hypernetwork input of one video
    step_zero, adapted, rank_max = 0.0, {}, 0
    for li in cfg.target_layers(n_layers):
        hn = LayerHyperNet(3 * hidden, dims, rank=rank, trunk_width=32, latent_size=16)
        with torch.no_grad():                  # V_m starts at zero (Eq. 9)
            for name, (A, B) in hn(h).items():
                step_zero = max(step_zero, float((B.transpose(1, 2) @ A).abs().max()))
        for proj in hn.proj_B.values():        # stand-in for trained weights
            nn.init.normal_(proj.weight, std=0.02)
        with torch.no_grad():
            st.updates[li] = upd = hn(h)
        layer = model.model.language_model.layers[li]
        for name, (A, B) in upd.items():
            x = torch.randn(1, tokens, dims[name][0])
            with torch.no_grad():
                adapted[(li, name)] = (x, _lora_linear(layer.map(name), x, A, B, st.scaling))
            dw = st.scaling * B[0].T @ A[0]
            rank_max = max(rank_max, int(torch.linalg.matrix_rank(dw)))

    merge_updates(model)
    with torch.no_grad():
        gap = max(float((model.model.language_model.layers[li].map(name)(x) - y).abs().max())
                  for (li, name), (x, y) in adapted.items())
    unmerge_updates(model)
    restored = max(float((p.detach() - before[n]).abs().max())
                   for n, p in model.named_parameters())
    return {"step_zero": step_zero, "merged_gap": gap, "restored_gap": restored,
            "rank": rank_max, "rank_limit": rank, "maps": len(adapted)}


# --------------------------------------------------------------------------- #
def main() -> int:
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    t0 = time.time()
    failures = []

    def line(label, value, want=None, ok=True, source="paper"):
        mark = "" if want is None else f"   {source} {want}   " + ("ok" if ok else "DIFFERENT")
        print(f"  {label:<41}{value}{mark}")
        if not ok:
            failures.append(label)

    print("WIRE CPU check: the package's functions against the paper\n")
    print("Sequence contraction at 256 frames (Sections 3.1 and 3.3, Appendix D)")
    full = contraction_lengths(16)
    line("decoder input (Section 3.1)", f"{full['positions']:,} positions, {full['visual']:,} visual")
    for k, want in PAPER["contracted"].items():
        r = contraction_lengths(k)
        line(f"after contraction, {k} tokens per frame", f"{r['contracted']:,} positions",
             f"{want:,}", r["contracted"] == want)

    p = trained_parameters()
    print("\nAdapted layers and trained parameters (Sections 3.4 and 4)")
    line("adapted layers", ", ".join(map(str, p["layers"])),
         "3, 7, ..., 35", p["layers"] == PAPER["layers"])
    line("adapted maps", f"{p['maps']}", f"{PAPER['maps']}", p["maps"] == PAPER["maps"])
    line("hypernetworks", f"{p['hypernetworks'] / 1e6:.1f}M", "763M",
         round(p["hypernetworks"] / 1e6) == round(PAPER["hypernetworks"] / 1e6))
    line("cross-attention readout", f"{p['cross_pool'] / 1e6:.1f}M", "16.8M",
         round(p["cross_pool"] / 1e5) == round(PAPER["cross_pool"] / 1e5))
    if p["compressor"] is None:
        line("compressor", "counted when the Mamba2 kernels are installed")
    else:
        total = p["hypernetworks"] + p["cross_pool"] + p["compressor"]
        line("compressor", f"{p['compressor'] / 1e6:.1f}M", "203M",
             round(p["compressor"] / 1e6) == round(PAPER["compressor"] / 1e6))
        line("all trained modules", f"{total / 1e9:.2f}B", "0.98B",
             round(total / 1e7) == round(PAPER["total"] / 1e7))

    m = merge_check()
    print(f"\nGenerated updates on a small decoder ({m['maps']} maps, Sections 3.4 and 3.5)")
    line("update at initialization", f"largest entry {m['step_zero']:.1g}", "zero",
         m["step_zero"] == 0.0)
    line("rank of each update", f"at most {m['rank']}", f"at most r = {m['rank_limit']}",
         m["rank"] <= m["rank_limit"], source="expected")
    line("merged maps against adapted maps", f"largest difference {m['merged_gap']:.1e}",
         "equal", m["merged_gap"] < 1e-4, source="expected")
    line("weights after unmerging", f"largest difference {m['restored_gap']:.1e}",
         "restored", m["restored_gap"] < 1e-5, source="expected")

    if failures:
        print(f"\nDifferent from the paper: {', '.join(failures)}")
        return 1
    print(f"\nAll checks passed in {time.time() - t0:.1f} s.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
