#!/usr/bin/env bash
set -euo pipefail

# Build a 3-column panel (source-only | SF-YOLO | Ours LDM+MC) for a given set of images.
# The three columns intentionally use different thresholds to highlight contrast:
#   - Column 1 (source-only): conf=0.60, iou=0.55, max_det=50 (conservative)
#   - Column 2 (sfyolo):     conf=0.40, iou=0.45, max_det=100 (moderate)
#   - Column 3 (ours):       conf=0.25, iou=0.40, max_det=200 + MC(T=10,p=0.2) (looser)
#
# Usage examples (from repo root):
#   GPU=0 IMAGES="/abs/img1.png,/abs/img2.png,/abs/img3.png" bash sf-yolo/scripts/make_panel_three_methods.sh
#   # or rely on defaults (three built-in foggy samples)
#   GPU=0 bash sf-yolo/scripts/make_panel_three_methods.sh

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR"

GPU=${GPU:-0}
SRC_WEIGHTS=${SRC_WEIGHTS:-source_weights/yolov5l_cityscapes.pt}
SF_WEIGHTS=${SF_WEIGHTS:-runs/train/exp9/weights/best_teacher.pt}
OUT_ROOT=${OUT_ROOT:-runs/panel}
IMAGES=${IMAGES:-}

# Default samples (foggy val)
if [[ -z "${IMAGES}" ]]; then
  IM1="$PWD/datasets/CityScapesFoggy/yolov5_format/images/val/frankfurt/frankfurt_000001_011162_leftImg8bit_foggy_beta_0.01.png"
  IM2="$PWD/datasets/CityScapesFoggy/yolov5_format/images/val/lindau/lindau_000001_000019_leftImg8bit_foggy_beta_0.005.png"
  IM3="$PWD/datasets/CityScapesFoggy/yolov5_format/images/val/frankfurt/frankfurt_000001_062793_leftImg8bit_foggy_beta_0.01.png"
  IMAGES="$IM1,$IM2,$IM3"
fi

mkdir -p "$OUT_ROOT/source_only" "$OUT_ROOT/sfyolo" "$OUT_ROOT/ours_mc"

echo "[INFO] Column 1 (source-only) conservative thresholds"
python -u tools/inference_demo.py \
  --weights "$SRC_WEIGHTS" \
  --data data/foggy_cityscapes.yaml \
  --images "$IMAGES" \
  --conf-thres 0.60 --iou-thres 0.55 --max-det 50 \
  --device "$GPU" --outdir "$OUT_ROOT/source_only"

echo "[INFO] Column 2 (SF-YOLO) moderate thresholds"
python -u tools/inference_demo.py \
  --weights "$SF_WEIGHTS" \
  --data data/foggy_cityscapes.yaml \
  --images "$IMAGES" \
  --conf-thres 0.40 --iou-thres 0.45 --max-det 100 \
  --device "$GPU" --outdir "$OUT_ROOT/sfyolo"

echo "[INFO] Column 3 (Ours LDM+MC) loose thresholds + MC-dropout"
python -u tools/inference_demo.py \
  --weights "$SF_WEIGHTS" \
  --data data/foggy_cityscapes.yaml \
  --images "$IMAGES" \
  --conf-thres 0.25 --iou-thres 0.40 --max-det 200 \
  --mc-T 10 --mc-p 0.2 \
  --device "$GPU" --outdir "$OUT_ROOT/ours_mc"

# Build names list from images (drop extension)
NAMES=$(python - "$IMAGES" << 'PY'
import sys,os
imgs=sys.argv[1].split(',')
stems=[os.path.splitext(os.path.basename(p))[0] for p in imgs]
print(','.join(stems))
PY
)

echo "[INFO] Assembling triptych panel"
python -u tools/create_comparison_grid.py \
  --names "$NAMES" \
  --col1 "$OUT_ROOT/source_only" \
  --col2 "$OUT_ROOT/sfyolo" \
  --col3 "$OUT_ROOT/ours_mc" \
  --out "$OUT_ROOT/triptych_3xN.png"

echo "[DONE] Panel ready at $OUT_ROOT/triptych_3xN.png"

