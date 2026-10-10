#!/usr/bin/env bash
# Train WIRE on the frozen Qwen3-VL-8B-Instruct with the settings of a
# released model, and stop at its released step.
#
#   NGPU=6 bash scripts/train.sh 16 [extra arguments of wire.train.train]
#
#   k           16, 12, 8 or 6 (refined tokens per frame)
#   DATA        training jsonl (scripts/data/build_llavavideo.py), default
#               data/llavavideo/llavavideo_mcqboost.jsonl
#   VIDEO_ROOT  prefix for relative video paths in DATA (optional)
#   NGPU        GPUs on this node; global batch 120 = NGPU x 2 x accumulation,
#               so NGPU must divide 60
#   OUT         output directory, default checkpoints/wire-k<k>
#   DRY_RUN=1   print the command instead of running it
set -euo pipefail
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  sed -n '2,/^set -euo pipefail/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'
  exit 0
fi
K=${1:?usage: bash scripts/train.sh <k: 16|12|8|6> [args]}
shift
case "$K" in
  16) STOP=5400 ;;
  12) STOP=4700 ;;
  8)  STOP=5500 ;;
  6)  STOP=5500 ;;
  *)  echo "k must be 16, 12, 8 or 6" >&2; exit 2 ;;
esac
REPO=$(cd "$(dirname "$0")/.." && pwd)
DATA=${DATA:-$REPO/data/llavavideo/llavavideo_mcqboost.jsonl}
NGPU=${NGPU:-4}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

CMD=(torchrun --nproc_per_node "$NGPU" -m wire.train.train
     --data "$DATA" ${VIDEO_ROOT:+--video-root "$VIDEO_ROOT"}
     --global-batch 120 --micro-batch 2 --lr 3e-5 --max-length 16384
     --save-steps 100
     --k "$K" --output-dir "${OUT:-$REPO/checkpoints/wire-k$K}"
     --stop-at-step "$STOP" "$@")
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  echo "${CMD[*]}"
else
  "${CMD[@]}"
fi
