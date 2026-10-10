#!/usr/bin/env python
r"""Answer a question about a video with WIRE (one GPU).

    python scripts/answer.py --video my_video.mp4 \
        --question "What does the person do after opening the fridge?"

Defaults follow the paper: the model with 16 tokens per frame, 256 frames
sampled uniformly over the whole clip and greedy decoding (protocol:
wire/inference.py). The model downloads on first use. `--no-update` runs the
same model with the generated weight update switched off.

Models (`--model`):
  wire-k16 | wire-k12 | wire-k8 | wire-k6
                      released models (wire/models.py), or a local
                      checkpoint directory / Hub repo id
  stock               the uncompressed backbone (Qwen3-VL-8B-Instruct)
  pool<k>             training-free bilinear pooling to k tokens per frame
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, help="video file")
    ap.add_argument("--question", required=True, help="question about the video")
    ap.add_argument("--model", default="wire-k16", help="model name or checkpoint (see above)")
    ap.add_argument("--frames", type=int, default=256,
                    help="frames sampled uniformly over the clip")
    ap.add_argument("--max-new-tokens", type=int, default=128,
                    help="longest answer, in tokens")
    ap.add_argument("--no-update", action="store_true",
                    help="switch the generated weight update off")
    ap.add_argument("--device", default="cuda", help="torch device")
    args = ap.parse_args()

    from wire import controls
    from wire.inference import generate, prepare_inputs, video_message
    from wire.models import load

    model, processor, base_ctl = load(args.model, args.device)
    if base_ctl is None:                       # the stock backbone
        if args.no_update:
            raise SystemExit("--no-update needs a WIRE model")
        ctx = contextlib.nullcontext()
    else:
        ctl = dict(base_ctl)
        if args.no_update:
            ctl["no_update"] = True
        ctx = controls(model, **ctl)
    inputs = prepare_inputs(processor, video_message(
        args.video, args.question, nframes=args.frames), args.device)
    with ctx:
        print(generate(model, processor, inputs,
                       max_new_tokens=args.max_new_tokens))


if __name__ == "__main__":
    main()
