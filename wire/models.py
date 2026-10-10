"""The released models.

Each entry is a Hugging Face Hub repo id (placeholder until the weights are
uploaded) and the `WireConfig` it was trained with. Released repos also carry
the config as `wire_config.json`, so `load_wire("<repo id>")` needs neither.
"""

from __future__ import annotations

from .config import WireConfig

# TODO(release): replace the placeholders with the published repo ids.
HUB_ORG = "<HF_ORG>"

MODELS = {
    # k: refined tokens per frame kept in the decoder input.
    "wire-k16": dict(repo=f"{HUB_ORG}/WIRE-Qwen3-VL-8B-k16",
                     config=WireConfig(tokens_kept_per_frame=16)),
    "wire-k12": dict(repo=f"{HUB_ORG}/WIRE-Qwen3-VL-8B-k12",
                     config=WireConfig(tokens_kept_per_frame=12)),
    "wire-k8": dict(repo=f"{HUB_ORG}/WIRE-Qwen3-VL-8B-k8",
                    config=WireConfig(tokens_kept_per_frame=8)),
    "wire-k6": dict(repo=f"{HUB_ORG}/WIRE-Qwen3-VL-8B-k6",
                    config=WireConfig(tokens_kept_per_frame=6)),
}


def resolve(name_or_path: str):
    """(checkpoint, config) for a registry name, Hub repo id or local dir."""
    if name_or_path in MODELS:
        m = MODELS[name_or_path]
        return m["repo"], m["config"]
    return name_or_path, None


def load(name: str, device: str = "cuda"):
    """Load a model for evaluation by name.

    name   a registry name, local checkpoint directory or Hub repo id;
           "stock" for the unmodified Qwen3-VL-8B-Instruct; or "pool<k>"
           for training-free bilinear pooling to k tokens per frame.
    Returns (model, processor, base_controls). `base_controls` are the
    `wire.controls` switches the model always runs with (None for "stock").
    """
    import re

    import torch

    from .loading import load_wire
    if name == "stock":
        from transformers import AutoModelForImageTextToText, AutoProcessor
        base = WireConfig().base_model
        model = AutoModelForImageTextToText.from_pretrained(
            base, dtype=torch.bfloat16, attn_implementation="sdpa")
        return model.to(device).eval(), AutoProcessor.from_pretrained(base), None
    m = re.fullmatch(r"pool(\d+)", name)
    if m:
        cfg = WireConfig(tokens_kept_per_frame=int(m.group(1)))
        model, proc = load_wire(None, config=cfg, device=device)
        return model, proc, {"no_update": True, "scan_enabled": False}
    ckpt, cfg = resolve(name)
    model, proc = load_wire(ckpt, config=cfg, device=device)
    return model, proc, {}
