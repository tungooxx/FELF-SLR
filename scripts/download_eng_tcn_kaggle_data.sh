#!/usr/bin/env bash
set -euo pipefail
OUT="${1:-/content/eng_tcn_r1r2_data}"
REPO="${2:-/workspace/local-vlm/SLR/FELF-SLR}"
mkdir -p "$OUT"

# Bootstrap the frozen control dependency before any network call so a Kaggle
# permission failure never cascades into a misleading factorial_dev import error.
VENDOR_FACTORIAL="$REPO/vendor/eng_tcn_r1r2/factorial_dev.py"
FACTORIAL_DIR=/workspace/local-vlm/SLR/slr1_factorial
FACTORIAL_DST="$FACTORIAL_DIR/factorial_dev.py"
EXPECTED_FACTORIAL_SHA=e5171fdeea7bfbd851c3112a52ca5a05fd9db248a8f86c33fb833e9ef7453662
mkdir -p "$FACTORIAL_DIR"
if [[ ! -f "$FACTORIAL_DST" ]]; then
  cp "$VENDOR_FACTORIAL" "$FACTORIAL_DST"
fi
python - "$FACTORIAL_DST" "$EXPECTED_FACTORIAL_SHA" <<'PY'
import hashlib,sys
from pathlib import Path
p=Path(sys.argv[1]); expected=sys.argv[2]
got=hashlib.sha256(p.read_bytes()).hexdigest()
if got!=expected:
    raise SystemExit(f'factorial_dev SHA mismatch: {got} != {expected}')
print('factorial_dev_bootstrap=PASS', got)
PY

kaggle_cli() {
  if command -v kaggle >/dev/null 2>&1; then
    command kaggle "$@"
  else
    python -c 'from kaggle.cli import main; main()' "$@"
  fi
}

if [[ -n "${ENG_TCN_KAGGLE_DATASET:-}" ]]; then
  CANDIDATES=("$ENG_TCN_KAGGLE_DATASET")
else
  CANDIDATES=(
    "chuckies/felf-slr-eng-tcn-r1r2-wlasl100-dev-cache"
    "lordchucky/felf-slr-eng-tcn-r1r2-wlasl100-dev-cache"
    "tungdsxx/felf-slr-eng-tcn-r1r2-wlasl100-dev-cache"
  )
fi

TMPBASE="$OUT/.download_attempts"
rm -rf "$TMPBASE"
mkdir -p "$TMPBASE"
SELECTED=""
for slug in "${CANDIDATES[@]}"; do
  safe="${slug//\//__}"
  tmp="$TMPBASE/$safe"
  mkdir -p "$tmp"
  echo "Trying private Kaggle dataset: $slug"
  if kaggle_cli datasets download -d "$slug" -p "$tmp" --force; then
    SELECTED="$slug"
    find "$tmp" -mindepth 1 -maxdepth 1 -exec mv -t "$OUT" {} +
    break
  fi
  rm -rf "$tmp"
done
rm -rf "$TMPBASE"

if [[ -z "$SELECTED" ]]; then
  echo >&2 "ERROR: Kaggle token cannot access any ENG-TCN private mirror."
  echo >&2 "Expected access to one of: chuckies, lordchucky, tungdsxx."
  echo >&2 "Use a token from one of those project profiles or set ENG_TCN_KAGGLE_DATASET to an accessible mirror."
  exit 43
fi

echo "Selected Kaggle dataset: $SELECTED"

# Kaggle may return an outer archive. Expand only that layer, preserving the
# inner reproducibility archive when it is present.
find "$OUT" -maxdepth 1 -type f -name '*.zip' ! -name 'eng_tcn_r1r2_wlasl100_dev_cache.zip' -print0 | while IFS= read -r -d '' z; do
  python -m zipfile -e "$z" "$OUT"
done

python "$REPO/scripts/prepare_eng_tcn_colab_data.py" --repo "$REPO" --dataset-dir "$OUT"
