#!/usr/bin/env bash
set -euo pipefail
SLUG="${ENG_TCN_KAGGLE_DATASET:-chuckies/felf-slr-eng-tcn-r1r2-wlasl100-dev-cache}"
OUT="${1:-/content/eng_tcn_r1r2_data}"
REPO="${2:-/workspace/local-vlm/SLR/FELF-SLR}"
mkdir -p "$OUT"
python -m kaggle datasets download -d "$SLUG" -p "$OUT" --force
# Kaggle may wrap uploaded files in an outer zip. Expand only that download layer.
find "$OUT" -maxdepth 1 -type f -name '*.zip' ! -name 'eng_tcn_r1r2_wlasl100_dev_cache.zip' -print0 | while IFS= read -r -d '' z; do python -m zipfile -e "$z" "$OUT"; done
python "$REPO/scripts/prepare_eng_tcn_colab_data.py" --repo "$REPO" --dataset-dir "$OUT"
