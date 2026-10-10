"""Loading released WIRE checkpoints."""

from __future__ import annotations

import json
import os
from typing import Dict, Optional

import torch

from .config import CONFIG_NAME, WireConfig
from .modeling import attach_wire

WEIGHTS_NAME = "wire.safetensors"


def _resolve(path_or_repo: str) -> str:
    """Local directory, or a Hugging Face Hub repo id to download."""
    if os.path.isdir(path_or_repo):
        return path_or_repo
    from huggingface_hub import snapshot_download
    return snapshot_download(path_or_repo)


def remap_training_keys(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Map keys of a full training checkpoint to `model.wire` keys.

    Accepts the ms-swift checkpoints written by `train/` (prefix `wire.`)
    and the original research checkpoints (prefix `hyperhybrid_hypernets.`,
    with the compressor under `bimba`).
    """
    out = {}
    for k, v in sd.items():
        if k.startswith("wire."):
            out[k[len("wire."):]] = v
        elif k.startswith("hyperhybrid_hypernets."):
            rest = k[len("hyperhybrid_hypernets."):]
            head, _, tail = rest.partition(".")
            if head.isdigit():
                out[f"hypernets.{head}.{tail}"] = v
            elif head == "bimba":
                out[f"compressor.{tail}"] = v
            elif head == "cross_pool":
                out[f"cross_pool.{tail}"] = v
    return out


def read_wire_state_dict(checkpoint: str) -> Dict[str, torch.Tensor]:
    """State dict of `model.wire` from a release or a training checkpoint."""
    from safetensors.torch import load_file
    release = os.path.join(checkpoint, WEIGHTS_NAME)
    if os.path.isfile(release):
        return load_file(release)
    index = os.path.join(checkpoint, "model.safetensors.index.json")
    if os.path.isfile(index):
        with open(index) as f:
            wmap = json.load(f)["weight_map"]
        keys = [k for k in wmap if k.startswith(("wire.", "hyperhybrid_hypernets."))]
        sd = {}
        for shard in sorted({wmap[k] for k in keys}):
            part = load_file(os.path.join(checkpoint, shard))
            sd.update({k: part[k] for k in keys if wmap[k] == shard})
            del part
        return remap_training_keys(sd)
    single = os.path.join(checkpoint, "model.safetensors")
    if os.path.isfile(single):
        return remap_training_keys(load_file(single))
    raise FileNotFoundError(f"no WIRE weights found in {checkpoint}")


def load_wire_weights(model, checkpoint: str) -> None:
    sd = read_wire_state_dict(checkpoint)
    missing, unexpected = model.wire.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing[:8]} "
                           f"unexpected={unexpected[:8]}")


def load_wire(checkpoint: Optional[str], config: Optional[WireConfig] = None,
              base_model: Optional[str] = None, device: str = "cuda",
              dtype: torch.dtype = torch.bfloat16, attn_impl: str = "sdpa"):
    """Load the frozen backbone, attach WIRE and load the trained weights.

    checkpoint  release directory / Hub repo id (contains wire_config.json
                and wire.safetensors) or a training checkpoint directory.
                None leaves the modules at initialization (with
                `controls(no_update=True, scan_enabled=False)` this is the
                training-free bilinear pooling baseline).
    Returns (model, processor).
    """
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if checkpoint is not None:
        checkpoint = _resolve(checkpoint)
        if config is None and os.path.isfile(os.path.join(checkpoint, CONFIG_NAME)):
            config = WireConfig.load(checkpoint)
    if config is None:
        raise ValueError("pass `config` when the checkpoint has no "
                         f"{CONFIG_NAME}")
    base_model = base_model or config.base_model
    model = AutoModelForImageTextToText.from_pretrained(
        base_model, dtype=dtype, attn_implementation=attn_impl)
    attach_wire(model, config)
    if checkpoint is not None:
        load_wire_weights(model, checkpoint)
    # inference runs every WIRE module in the backbone's dtype
    model.wire.to(device, dtype)
    model.to(device).eval()
    processor = AutoProcessor.from_pretrained(base_model)
    return model, processor
