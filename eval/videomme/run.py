#!/usr/bin/env python
r"""Video-MME (no subtitles) with WIRE, 256 frames (protocol: wire/inference.py).

One process evaluates one model under one condition on a list of duration
splits. Predictions are appended to `<out>/preds_shard<i>.jsonl` (resumable);
`--report-only` merges all shards into `<out>/summary.json`.

Models (`--model`):
  wire-k16 | wire-k12 | wire-k8 | wire-k6
                      registry names (see wire/models.py), or a local
                      checkpoint directory / Hub repo id
  stock               the unmodified Qwen3-VL-8B-Instruct
  pool<k>             training-free bilinear pooling to k tokens per frame
                      (no scan, no generated update), e.g. pool16

Conditions (`--condition`, WIRE models only):
  own                 the model as trained
  no_update           generated weight updates switched off (Delta W = 0)
  zero_context        no visual token in the decoder input; the video enters
                      only through the generated weights
  frame_drop          a random half of the frames removed from the decoder
                      input (the hypernetwork still sees all of them)
  blind               neither visual tokens nor an update: the backbone sees
                      the question and the frame timestamps only

Example:
  python eval/videomme/run.py --model wire-k16 --condition own \
      --video-root data/videomme/videos --durations short,medium,long \
      --output-dir results/videomme/wire-k16_own
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import glob
import json
import os
import re
import sys
import time
from collections import defaultdict


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

CONDITIONS = {
    "own": {},
    "no_update": {"no_update": True},
    "zero_context": {"zero_context": True},
    "frame_drop": {"frame_drop": 0.5},
    # text and frame timestamps only: neither visual tokens nor an update
    "blind": {"zero_context": True, "no_update": True},
}

# --------------------------------------------------------------------------- #
# prompt and answer extraction (lmms-eval `videomme`)
# --------------------------------------------------------------------------- #
PRE_PROMPT = ("Select the best answer to the following multiple-choice question "
              "based on the video. Respond with only the letter (A, B, C, or D) "
              "of the correct option.\n")
POST_PROMPT = "\nThe best answer is:"
_ANSWER_PREFIXES = [
    "The best answer is", "The correct answer is", "The answer is",
    "The answer", "The best option is", "The correct option is",
    "Best answer:", "Best option:", "Answer:", "Option:",
    "The correct answer", "The correct option",
]


def build_prompt(question: str, options) -> str:
    return f"{PRE_PROMPT}{question}\n" + "\n".join(map(str, options)) + POST_PROMPT


def extract_answer_letter(text: str) -> str:
    """Port of lmms-eval's `extract_characters_regex`."""
    s = (text or "").strip()
    for p in _ANSWER_PREFIXES:
        s = s.replace(p, "")
    s = s.strip()
    if len(s.split()) > 10 and not re.search("[ABCD]", s):
        return ""
    m = re.search(r"[ABCD]", s)
    return m.group(0) if m else ""


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load_videomme(dataset: str):
    """Question table, read from the parquet files of the Hub dataset."""
    import pandas as pd
    from huggingface_hub import hf_hub_download, list_repo_files
    files = sorted(f for f in list_repo_files(dataset, repo_type="dataset")
                   if f.endswith(".parquet"))
    return pd.concat([pd.read_parquet(hf_hub_download(dataset, f, repo_type="dataset"))
                      for f in files], ignore_index=True)


def resolve_video(root: str, video_id: str):
    exts = (".mp4", ".mkv", ".webm", ".avi", ".mov", ".MP4")
    for d in (root, os.path.join(root, "data")):   # the zips unpack into data/
        for ext in exts:
            p = os.path.join(d, video_id + ext)
            if os.path.isfile(p):
                return p
    for ext in exts:
        hits = glob.glob(os.path.join(root, "**", video_id + ext), recursive=True)
        if hits:
            return hits[0]
    return None


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def load_preds(out_dir: str) -> list:
    recs = {}
    for fp in sorted(glob.glob(os.path.join(out_dir, "preds_shard*.jsonl"))):
        with open(fp) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    recs[r["question_id"]] = r
    return list(recs.values())


def summarize(records: list) -> dict:
    tot, by_dur, by_task = [0, 0], defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
    for r in records:
        ok = int(bool(r["correct"]))
        for c in (tot, by_dur[r["duration"]], by_task[r["task_type"]]):
            c[0] += ok
            c[1] += 1

    def acc(c):
        return {"acc": round(100.0 * c[0] / c[1], 2) if c[1] else 0.0, "n": c[1]}
    return {"overall": acc(tot),
            "by_duration": {k: acc(v) for k, v in sorted(by_dur.items())},
            "by_task_type": {k: acc(v) for k, v in sorted(by_task.items())}}


def report(out_dir: str) -> dict:
    s = summarize(load_preds(out_dir))
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(s, f, indent=2)
    durs = "  ".join(f"{k} {v['acc']:.2f} (n={v['n']})" for k, v in s["by_duration"].items())
    print(f"{out_dir}: overall {s['overall']['acc']:.2f} (n={s['overall']['n']})  {durs}")
    return s


def run(args) -> None:
    from wire import controls
    from wire.inference import generate, prepare_inputs, video_message
    from wire.models import load

    df = load_videomme(args.dataset)
    df = df[df["duration"].isin(args.durations.split(","))]
    if args.limit:
        df = df.iloc[:args.limit]
    df = df.iloc[args.shard_id::args.num_shards]

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"preds_shard{args.shard_id}.jsonl")
    done = {r["question_id"] for r in load_preds(args.output_dir)}
    print(f"[shard {args.shard_id}] {len(df)} questions, {len(done)} already done", flush=True)

    model, processor, base_ctl = load(args.model, args.device)
    ctl = dict(CONDITIONS[args.condition])
    if ctl.get("frame_drop"):
        ctl["frame_drop_seed"] = args.seed
    if base_ctl is None:
        if ctl:
            raise ValueError(f"condition {args.condition!r} needs a WIRE model")
        ctx = contextlib.nullcontext
    else:
        ctl.update(base_ctl)
        ctx = functools.partial(controls, model, **ctl)

    n = n_ok = n_missing = 0
    t0 = time.time()
    with open(out_path, "a", buffering=1) as fout:
        for _, row in df.iterrows():
            if row["question_id"] in done:
                continue
            video = resolve_video(args.video_root, row["videoID"])
            if video is None:
                n_missing += 1
                print(f"missing video {row['videoID']}", flush=True)
                continue
            try:
                inputs = prepare_inputs(processor, video_message(
                    video, build_prompt(row["question"], row["options"]),
                    nframes=args.nframes), args.device)
                with ctx():
                    text = generate(model, processor, inputs,
                                    max_new_tokens=args.max_new_tokens)
            except Exception as e:  # keep long runs alive; counted as wrong
                print(f"ERROR {row['question_id']}: {type(e).__name__}: {e}", flush=True)
                text = ""
            pred, gold = extract_answer_letter(text), str(row["answer"]).strip().upper()
            rec = {k: row[k] for k in ("question_id", "videoID", "duration",
                                       "domain", "sub_category", "task_type")
                   if k in row}
            rec.update(pred=pred, gold=gold, correct=pred == gold, raw=text)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
            n_ok += pred == gold
            if n % args.log_every == 0:
                print(f"[shard {args.shard_id}] {n} done | acc {100 * n_ok / n:.2f} | "
                      f"{n / (time.time() - t0):.2f} q/s", flush=True)
    print(f"[shard {args.shard_id}] finished: {n} answered, {n_missing} missing videos", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="wire-k16")
    ap.add_argument("--condition", default="own", choices=sorted(CONDITIONS))
    ap.add_argument("--video-root", default="data/videomme/videos")
    ap.add_argument("--dataset", default="lmms-lab/Video-MME")
    ap.add_argument("--durations", default="short,medium,long")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--nframes", type=int, default=256)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0, help="frame_drop seed")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    if not args.report_only:
        run(args)
    if args.report_only or args.num_shards == 1:
        report(args.output_dir)


if __name__ == "__main__":
    main()
