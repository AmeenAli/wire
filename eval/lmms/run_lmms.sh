#!/usr/bin/env bash
# Run one lmms-eval benchmark with a WIRE model on one GPU.
#
#   bash eval/lmms/run_lmms.sh <model> <task> [condition] [extra lmms-eval args]
#
#   model      wire-k16 | wire-k12 | wire-k8 | wire-k6 | stock | pool<k> |
#              <checkpoint dir or Hub repo id>
#   task       nextqa_mc_test | perceptiontest_val_mc | mvbench | tomato |
#              videomme_w_subtitle | longvideobench_val_v | mlvu_test | lvbench
#   condition  own (default) | no_update | zero_context | frame_drop
#
# Needs lmms-eval 0.4.0 installed with the MLVU patch applied (README).
# Environment: LMMS_OUT (default results/lmms), LVBENCH_DIR (default
# $HF_HOME/lvbench), CUDA_VISIBLE_DEVICES.
# lmms-eval cannot resume; a crashed run restarts from scratch, so the
# script retries up to RETRIES times (default 3).
set -euo pipefail

MODEL=$1; TASK=$2; COND=${3:-own}; shift $(( $# < 3 ? $# : 3 ))
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
OUT=${LMMS_OUT:-$REPO/results/lmms}/${MODEL//\//_}_${COND}_${TASK}
export LVBENCH_DIR=${LVBENCH_DIR:-${HF_HOME:-$HOME/.cache/huggingface}/lvbench}
sed "s#@LVBENCH_DIR@#$LVBENCH_DIR#g" "$HERE/tasks/lvbench/lvbench.yaml.in" > "$HERE/tasks/lvbench/lvbench.yaml"

export PYTHONPATH="$HERE:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export LMMS_EVAL_PLUGINS=wire_lmms
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for attempt in $(seq 1 "${RETRIES:-3}"); do
  echo "[run_lmms] attempt $attempt: $MODEL / $COND / $TASK -> $OUT"
  python -m lmms_eval \
    --model wire \
    --model_args "model=$MODEL,condition=$COND,nframes=256" \
    --tasks "$TASK" --include_path "$HERE/tasks" \
    --batch_size 1 --log_samples --output_path "$OUT" "$@" && break
done
python "$HERE/check_format.py" "$OUT"
