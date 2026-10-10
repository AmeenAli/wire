#!/usr/bin/env python
"""Rescore MLVU (test, MCQ) samples with the fixed answer extractor.

lmms-eval 0.4's MLVU extractor returns the raw response when it has no ")",
so answers such as "D." never match "D". eval/lmms/patches/ fixes this for
new runs; this script rescores the `*_samples_mlvu_test.jsonl` of runs made
without the patch (macro average over the nine task types, as lmms-eval).

    python scripts/paper/rescore_mlvu.py results/lmms/wire-k16_own_mlvu_test [...]
"""

import argparse
import glob
import json
import os
import re
from collections import defaultdict

TASK_TYPES = ["anomaly_reco", "count", "ego", "needleQA", "order", "plotQA",
              "sportsQA", "topic_reasoning", "tutorialQA"]


def extract(s: str) -> str:
    """Identical to the patched `extract_characters_regex`."""
    s = s.strip()
    if ")" in s:
        i = s.index(")")
        pred = s[i - 1:i]
        if re.fullmatch(r"[A-Za-z]", pred):
            return pred.upper()
    for pat in (r"^\(?\s*([A-Za-z])\s*[\.\):,\-]", r"^\(?\s*([A-Za-z])\s*$"):
        m = re.match(pat, s)
        if m:
            return m.group(1).upper()
    m = re.search(r"\b([A-Ha-h])\b", s)
    return m.group(1).upper() if m else s


def rescore(d):
    files = glob.glob(os.path.join(d, "**", "*_samples_mlvu_test*.jsonl"), recursive=True)
    score = defaultdict(lambda: [0, 0])
    for fp in files:
        for line in open(fp):
            r = json.loads(line)
            meta = r["mlvu_percetion_score"]
            resp = r["filtered_resps"][0] if r.get("filtered_resps") else r["resps"][0][0]
            score[meta["task_type"]][0] += extract(resp) == meta["answer"]
            score[meta["task_type"]][1] += 1
    per_type = {t: 100 * score[t][0] / score[t][1] if score[t][1] else 0.0
                for t in TASK_TYPES}
    return sum(per_type.values()) / len(TASK_TYPES), per_type, sum(v[1] for v in score.values())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+", help="lmms-eval output directories")
    for d in ap.parse_args().dirs:
        avg, per_type, n = rescore(d)
        print(f"{d}: MLVU {avg:.2f} (n={n})  " +
              "  ".join(f"{t} {v:.1f}" for t, v in per_type.items()))


if __name__ == "__main__":
    main()
