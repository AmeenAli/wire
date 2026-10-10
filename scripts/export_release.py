#!/usr/bin/env python
r"""Export a training checkpoint as a release directory.

Writes `wire.safetensors` (the WIRE modules in bfloat16, the dtype they run
in at inference) and `wire_config.json`. Accepts checkpoints written by
`wire.train.train` and the original research checkpoints.

    python scripts/export_release.py checkpoints/wire-k16/checkpoint-<step> \
        release/WIRE-Qwen3-VL-8B-k16 --model wire-k16
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from wire.config import WireConfig  # noqa: E402
from wire.loading import WEIGHTS_NAME, read_wire_state_dict  # noqa: E402
from wire.models import MODELS  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("out")
    ap.add_argument("--model", choices=sorted(MODELS),
                    help="registry entry whose config to write (needed for "
                         "checkpoints without wire_config.json)")
    args = ap.parse_args()
    from safetensors.torch import save_file

    if os.path.isfile(os.path.join(args.checkpoint, "wire_config.json")):
        cfg = WireConfig.load(args.checkpoint)
    elif args.model:
        cfg = MODELS[args.model]["config"]
    else:
        raise SystemExit("no wire_config.json in the checkpoint: pass --model")
    sd = {k: v.to(torch.bfloat16).contiguous()
          for k, v in read_wire_state_dict(args.checkpoint).items()}
    os.makedirs(args.out, exist_ok=True)
    save_file(sd, os.path.join(args.out, WEIGHTS_NAME), metadata={"format": "pt"})
    cfg.save(args.out)
    n = sum(v.numel() for v in sd.values())
    print(f"{len(sd)} tensors, {n / 1e9:.3f}B parameters -> {args.out}")


if __name__ == "__main__":
    main()
