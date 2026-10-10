#!/usr/bin/env python
"""Share of answers in option-letter format in an lmms-eval run.

lmms-eval's multiple-choice parser falls back to a random option when the
response holds no letter, so a run with many off-format answers is partly a
coin flip. This matters for the `no_update` condition, where some models stop
answering with a bare letter. Runs above 2% off-format are flagged.

    python eval/lmms/check_format.py <results_dir> [...]
"""
import argparse
import glob
import json
import os
import re

LETTER = re.compile(r"^\(?\s*(?:answer\s*[:\-]?\s*)?[A-Ea-e]\b\s*[\).:,]?\s*$")
LETTER_PREFIX = re.compile(r"^\(?\s*[A-Ea-e]\s*[\).:]")


def rate(d):
    n = bad = 0
    for f in glob.glob(os.path.join(d, "**", "*_samples_*.jsonl"), recursive=True):
        for line in open(f):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            resp = r.get("resps") or r.get("filtered_resps") or []
            while isinstance(resp, list) and resp:
                resp = resp[0]
            if not isinstance(resp, str):
                continue
            n += 1
            t = resp.strip()
            bad += not (LETTER.match(t) or LETTER_PREFIX.match(t))
    return n, bad


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+", help="lmms-eval output directories")
    for d in ap.parse_args().dirs:
        n, bad = rate(d)
        if n == 0:
            print(f"{d}: no samples found")
            continue
        flag = "  <-- more than 2% off-format" if bad / n > 0.02 else ""
        print(f"{d}: in-format {100 * (n - bad) / n:.2f}% ({n - bad}/{n}){flag}")


if __name__ == "__main__":
    main()
