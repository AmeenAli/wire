#!/usr/bin/env bash
# Download and unpack the 900 Video-MME videos (20 zip chunks, ~100 GB).
# Usage: bash eval/videomme/download.sh [target_dir]   (default data/videomme)
# Resumable: finished chunks are marked and skipped.
set -euo pipefail

BASE="https://huggingface.co/datasets/lmms-lab/Video-MME/resolve/main"
ROOT="${1:-data/videomme}"
mkdir -p "$ROOT/videos" "$ROOT/zips"

for i in $(seq -w 1 20); do
  mark="$ROOT/.done_$i"
  [[ -f "$mark" ]] && continue
  zip="$ROOT/zips/videos_chunked_$i.zip"
  echo "[chunk $i] downloading"
  curl -fL --retry 5 --retry-delay 10 -C - -o "$zip" "$BASE/videos_chunked_$i.zip"
  unzip -q -o "$zip" -d "$ROOT/videos"
  rm -f "$zip"
  touch "$mark"
done
echo "done: $(find "$ROOT/videos" -type f | wc -l) videos in $ROOT/videos"
