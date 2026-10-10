#!/usr/bin/env bash
# Table 1 of the paper: the released models on the nine video benchmarks at
# 256 frames (one GPU; REPRODUCING.md lists the paper's numbers).
#
#   bash reproduce/main_results.sh
#   DRY_RUN=1 bash reproduce/main_results.sh          # print the commands
#   STAGES=videomme MODELS=wire-k16 bash reproduce/main_results.sh
#
#   STAGES         videomme: Video-MME without subtitles (eval/videomme)
#                  lmms: the other eight benchmarks (eval/lmms, lmms-eval)
#   MODELS         default: wire-k16 wire-k12 wire-k8 wire-k6
#   TASKS          lmms-eval tasks, default: the other eight of Table 1
#   VIDEOMME_ROOT  Video-MME videos, default data/videomme/videos
# Results go to results/videomme/ and results/lmms/.
set -euo pipefail
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  sed -n '2,/^set -euo pipefail/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'
  exit 0
fi
if [[ $# -gt 0 ]]; then
  echo "unknown argument: $1 (settings are environment variables; see --help)" >&2
  exit 2
fi
REPO=$(cd "$(dirname "$0")/.." && pwd)
STAGES=${STAGES:-videomme lmms}
MODELS=${MODELS:-wire-k16 wire-k12 wire-k8 wire-k6}
TASKS=${TASKS:-nextqa_mc_test perceptiontest_val_mc mvbench tomato videomme_w_subtitle longvideobench_val_v mlvu_test lvbench}
VIDEOMME_ROOT=${VIDEOMME_ROOT:-$REPO/data/videomme/videos}

run() {
  echo "+ $*"
  if [[ "${DRY_RUN:-0}" != 1 ]]; then "$@"; fi
}

for stage in $STAGES; do
  case "$stage" in
    videomme)
      for m in $MODELS; do
        run python "$REPO/eval/videomme/run.py" --model "$m" \
            --video-root "$VIDEOMME_ROOT" --output-dir "$REPO/results/videomme/${m}_own"
      done ;;
    lmms)
      for m in $MODELS; do
        for t in $TASKS; do
          run bash "$REPO/eval/lmms/run_lmms.sh" "$m" "$t"
        done
      done ;;
    *) echo "unknown stage: $stage (videomme, lmms)" >&2; exit 2 ;;
  esac
done
