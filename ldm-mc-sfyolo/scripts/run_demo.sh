#!/usr/bin/env bash
set -euo pipefail

# End-to-end pipeline for Foggy Cityscapes → YOLO → Train → Visualize.
# Assumes you have either ZIPs (preferred) or extracted dirs.
# Optional env:
#   FOGGY_ZIP, GTFINE_ZIP  (or FOGGY_DIR, GTFINE_DIR)
#   SOURCE_WEIGHTS (default: sf-yolo/source_weights/yolov5l_cityscapes.pt)
#   DEVICE (default: 0)

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
SCRIPTS="$ROOT_DIR/scripts"
TOOLS="$ROOT_DIR/tools"
DATA_CFG="$ROOT_DIR/data/foggy_cityscapes.yaml"
OUTDIR="$ROOT_DIR/datasets/CityScapesFoggy/yolov5_format"
SOURCE_WEIGHTS="${SOURCE_WEIGHTS:-$ROOT_DIR/source_weights/yolov5l_cityscapes.pt}"
DEVICE="${DEVICE:-0}"

echo "[1/4] Building YOLO dataset at: $OUTDIR"
if [[ -n "${FOGGY_ZIP:-}" || -n "${GTFINE_ZIP:-}" || -n "${FOGGY_DIR:-}" || -n "${GTFINE_DIR:-}" ]]; then
  bash "$SCRIPTS/prepare_foggy_cityscapes.sh"
else
  if [[ -f "$ROOT_DIR/datasets/_zips/leftImg8bit_trainvaltest_foggy.zip" && -f "$ROOT_DIR/datasets/_zips/gtFine_trainvaltest.zip" ]]; then
    FOGGY_ZIP="$ROOT_DIR/datasets/_zips/leftImg8bit_trainvaltest_foggy.zip" \
    GTFINE_ZIP="$ROOT_DIR/datasets/_zips/gtFine_trainvaltest.zip" \
    bash "$SCRIPTS/prepare_foggy_cityscapes.sh"
  else
    echo "[ERROR] Provide FOGGY_ZIP/GTFINE_ZIP or place ZIPs under datasets/_zips/." >&2
    exit 1
  fi
fi

echo "[2/4] Verifying source weights: $SOURCE_WEIGHTS"
if [[ ! -f "$SOURCE_WEIGHTS" ]]; then
  echo "[ERROR] Missing source weights: $SOURCE_WEIGHTS" >&2
  echo "        Place yolov5l_cityscapes.pt under sf-yolo/source_weights/ or set SOURCE_WEIGHTS env."
  exit 1
fi

echo "[3/4] Starting training (LDM augmenter by default)"
python "$ROOT_DIR/run_adaptation.py" \
  --epochs 60 --batch-size 16 \
  --data "$DATA_CFG" \
  --weights "$SOURCE_WEIGHTS" \
  --ta_method ldm --ldm_strength 0.30 --ldm_steps 10 --ldm_prob 0.5 \
  --ldm_cache_dir "$ROOT_DIR/datasets/CityScapesFoggy/ldm_cache" \
  --SSM_alpha 0.5 --device "$DEVICE"

echo "[4/4] Visualizing a few images"
best_ckpt=$(ls -td "$ROOT_DIR"/runs/*/weights/best_teacher.pt 2>/dev/null | head -n1 || true)
if [[ -n "$best_ckpt" ]]; then
  python "$TOOLS/visualize_augmentation.py" --weights "$best_ckpt" --data "$DATA_CFG" --num-images 10 --outdir "$ROOT_DIR/runs/vis_compare"
  echo "[OK] Visualization written to runs/vis_compare"
else
  echo "[WARN] Could not find best_teacher.pt for visualization."
fi

echo "[DONE] Pipeline complete."
