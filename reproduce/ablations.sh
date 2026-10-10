#!/usr/bin/env bash
# Section 5.1 and Appendix E of the paper on Video-MME without subtitles at
# 256 frames (one GPU; REPRODUCING.md lists the paper's numbers).
#
#   bash reproduce/ablations.sh
#   DRY_RUN=1 bash reproduce/ablations.sh             # print the commands
#   STAGES=update MODELS="wire-k16 wire-k6" bash reproduce/ablations.sh
#
#   STAGES         pooling: training-free bilinear pooling with the same
#                           contraction, 16 tokens per frame
#                  backbone: the uncompressed Qwen3-VL-8B-Instruct, short videos
#                  update: each model with its generated update switched off
#   MODELS         models of the update stage, default: wire-k16
#   VIDEOMME_ROOT  Video-MME videos, default data/videomme/videos
# Compare with the runs of reproduce/main_results.sh (results/videomme/).
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
STAGES=${STAGES:-pooling backbone update}
MODELS=${MODELS:-wire-k16}
VIDEOMME_ROOT=${VIDEOMME_ROOT:-$REPO/data/videomme/videos}
OUT=$REPO/results/videomme

run() {
  echo "+ $*"
  if [[ "${DRY_RUN:-0}" != 1 ]]; then "$@"; fi
}

for stage in $STAGES; do
  case "$stage" in
    pooling)
      run python "$REPO/eval/videomme/run.py" --model pool16 \
          --video-root "$VIDEOMME_ROOT" --output-dir "$OUT/pool16" ;;
    backbone)
      run python "$REPO/eval/videomme/run.py" --model stock --durations short \
          --video-root "$VIDEOMME_ROOT" --output-dir "$OUT/stock_short" ;;
    update)
      for m in $MODELS; do
        run python "$REPO/eval/videomme/run.py" --model "$m" --condition no_update \
            --video-root "$VIDEOMME_ROOT" --output-dir "$OUT/${m}_no_update"
      done ;;
    *) echo "unknown stage: $stage (pooling, backbone, update)" >&2; exit 2 ;;
  esac
done
