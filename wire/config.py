"""Configuration for WIRE.

`WireConfig` holds everything that determines the architecture and is saved
next to a checkpoint as `wire_config.json`. Evaluation-time switches that do
not change the parameters (the ablations) are set on the attached model with
`wire.modeling.controls(...)`.
"""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass
from typing import List, Optional

CONFIG_NAME = "wire_config.json"


@dataclass
class WireConfig:
    # --- sequence contraction -------------------------------------------------
    # k: refined tokens kept per frame in the decoder input (paper: k).
    tokens_kept_per_frame: int = 16
    # eta: nominal visual tokens per frame at the 114,688-pixel budget.
    tokens_per_frame: int = 56
    # True for the k = 0 model: no visual token reaches the decoder; the video
    # enters only through the generated weights.
    zero_context: bool = False

    # --- selective scan compressor -------------------------------------------
    scan_layers: int = 1
    scan_d_state: int = 64
    scan_d_conv: int = 4
    scan_expand: int = 2
    scan_headdim: int = 128
    # Question embeddings bracket the scanned sequence (Eq. 4).
    scan_with_text: bool = True
    # Std of the scan / cross-attention output projections at init
    # (0 = exact identity at step 0).
    out_proj_init_std: float = 3e-4

    # --- hypernetworks --------------------------------------------------------
    # Adapted layers; None = every `layer_stride`-th layer ending at the last
    # (36 layers, stride 4 -> 3, 7, ..., 35).
    layers: Optional[List[int]] = None
    layer_stride: int = 4
    rank: int = 64
    alpha: int = 128
    trunk_width: int = 512
    latent_size: int = 256
    adapt_mlp: bool = True
    # Cross-attention readout of the refined full-resolution rows (Eq. 6).
    cross_pool_dim: int = 1024
    cross_pool_heads: int = 8
    # Input-dropout recipe: condition on the mean of the refined queries
    # instead of the residual stream at visual positions (Sec. 3.6).
    cond_on_queries: bool = False

    base_model: str = "Qwen/Qwen3-VL-8B-Instruct"

    @property
    def scaling(self) -> float:
        return self.alpha / self.rank

    def target_layers(self, num_hidden_layers: int) -> List[int]:
        if self.layers:
            return sorted(set(self.layers))
        s = self.layer_stride
        return list(range(s - 1, num_hidden_layers, s))

    # --- (de)serialization ----------------------------------------------------
    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, CONFIG_NAME), "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "WireConfig":
        names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})

    @classmethod
    def load(cls, path: str) -> "WireConfig":
        if os.path.isdir(path):
            path = os.path.join(path, CONFIG_NAME)
        with open(path) as f:
            return cls.from_dict(json.load(f))

