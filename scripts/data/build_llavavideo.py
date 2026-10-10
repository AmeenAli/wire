#!/usr/bin/env python
r"""Build the training jsonl from LLaVA-Video-178K.

1. Every `*_qa_processed.json` (open-ended and multiple-choice QA; the
   caption files are not used) -> one record per video with its first QA
   turn, in the ms-swift messages format (1,217,025 records; `llava_hound`
   has no videos and drops out).
2. Multiple-choice records (answer starts with "A." / "B)" / ...; 215,682)
   are added twice more, and the result is shuffled: 1,648,389 records.

    python scripts/data/build_llavavideo.py --data-root data/LLaVA-Video-178K \
        --out data/llavavideo/llavavideo_mcqboost.jsonl

`--data-root` is the extracted dataset: one directory per bucket
(`0_30_s_youtube_v0_1`, ...) holding the json files and the videos (either
under `<bucket>/videos/` or directly in the bucket).
"""

import argparse
import glob
import json
import os
import random
import re

MARKER = re.compile(r"\n?(?:<image>|<video>)\n?")
MCQ = re.compile(r"^[A-Z][.):]")


def records(root):
    n_missing = 0
    for bucket in sorted(os.listdir(root)):
        bdir = os.path.join(root, bucket)
        if not os.path.isdir(bdir):
            continue
        n = 0
        for jf in sorted(glob.glob(os.path.join(bdir, "*_qa_processed.json"))):
            try:
                recs = json.load(open(jf))
            except ValueError:
                continue
            for r in recs:
                rel, conv = r.get("video", ""), r.get("conversations", [])
                if len(conv) < 2 or not rel:
                    continue
                path = next((p for p in (os.path.join(bdir, "videos", rel),
                                         os.path.join(bdir, rel))
                             if os.path.exists(p)), None)
                if path is None:
                    n_missing += 1
                    continue
                q = MARKER.sub("", conv[0]["value"]).strip()
                a = conv[1]["value"].strip()
                if q and a:
                    n += 1
                    yield {"messages": [{"role": "user", "content": f"<video>{q}"},
                                        {"role": "assistant", "content": a}],
                           "videos": [os.path.abspath(path)]}
        print(f"[{bucket}] {n} records ({n_missing} missing videos so far)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mcq-copies", type=int, default=2,
                    help="extra copies of every multiple-choice record")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--from-jsonl", help="skip step 1: boost an existing "
                    "step-1 jsonl instead of scanning --data-root")
    args = ap.parse_args()

    if args.from_jsonl:
        base = [json.loads(line) for line in open(args.from_jsonl)]
    else:
        base = list(records(args.data_root))
    mcq = [r for r in base if MCQ.match(r["messages"][1]["content"])]
    rows = base + mcq * args.mcq_copies
    random.Random(args.seed).shuffle(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(base)} records, {len(mcq)} multiple-choice, "
          f"{len(rows)} written to {args.out}")


if __name__ == "__main__":
    main()
