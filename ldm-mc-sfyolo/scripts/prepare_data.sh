#!/usr/bin/env bash
set -euo pipefail

# Prepare Foggy Cityscapes YOLO dataset from official zips or extracted dirs.
# Usage examples:
#  FOGGY_ZIP=/path/leftImg8bit_foggy_trainvaltest.zip GTFINE_ZIP=/path/gtFine_trainvaltest.zip bash scripts/prepare_foggy_cityscapes.sh
#  or if already extracted:
#  FOGGY_DIR=/data/leftImg8bit_foggy GTFINE_DIR=/data/gtFine bash scripts/prepare_foggy_cityscapes.sh

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
OUT_ROOT="$ROOT_DIR/datasets/CityScapesFoggy"
YOLO_OUT="$OUT_ROOT/yolov5_format"
RAW_DIR="$OUT_ROOT/raw"
mkdir -p "$RAW_DIR"

echo "[INFO] Output YOLO root: $YOLO_OUT"

if [[ -n "${FOGGY_DIR:-}" && -d "$FOGGY_DIR" ]]; then
  echo "[INFO] Using extracted foggy dir: $FOGGY_DIR"
  FOGGY_EXTRACT="$FOGGY_DIR"
elif [[ -n "${FOGGY_ZIP:-}" && -f "$FOGGY_ZIP" ]]; then
  echo "[INFO] Extracting foggy zip: $FOGGY_ZIP"
  unzip -q -o "$FOGGY_ZIP" -d "$RAW_DIR"
  FOGGY_EXTRACT=$(find "$RAW_DIR" -maxdepth 2 -type d -name 'leftImg8bit_foggy' | head -n1)
else
  echo "[ERROR] Provide FOGGY_DIR (extracted) or FOGGY_ZIP (zip path)." >&2
  exit 1
fi

if [[ -n "${GTFINE_DIR:-}" && -d "$GTFINE_DIR" ]]; then
  echo "[INFO] Using extracted gtFine dir: $GTFINE_DIR"
  GT_EXTRACT="$GTFINE_DIR"
elif [[ -n "${GTFINE_ZIP:-}" && -f "$GTFINE_ZIP" ]]; then
  echo "[INFO] Extracting gtFine zip: $GTFINE_ZIP"
  unzip -q -o "$GTFINE_ZIP" -d "$RAW_DIR"
  GT_EXTRACT=$(find "$RAW_DIR" -maxdepth 2 -type d -name 'gtFine' | head -n1)
else
  echo "[ERROR] Provide GTFINE_DIR (extracted) or GTFINE_ZIP (zip path)." >&2
  exit 1
fi

echo "[INFO] Foggy at: $FOGGY_EXTRACT"
echo "[INFO] gtFine at: $GT_EXTRACT"

python "$ROOT_DIR/tools/make_foggy_cityscapes_yolo.py" \
  --foggy-root "$FOGGY_EXTRACT" \
  --gt-root "$GT_EXTRACT" \
  --out-root "$YOLO_OUT" \
  --splits train val \
  --link-images

echo "[OK] YOLO dataset built at: $YOLO_OUT"

