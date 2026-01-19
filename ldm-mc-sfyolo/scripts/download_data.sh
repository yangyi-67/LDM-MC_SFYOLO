#!/usr/bin/env bash
set -euo pipefail

# Download Cityscapes gtFine + Foggy zips using a login session.
# Safer to pass credentials via env vars:
#   CS_USER='you@example.com' CS_PASS='your_password' bash sf-yolo/scripts/download_cityscapes.sh
# Outputs to: sf-yolo/datasets/_zips/

UA="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0 Safari/537.36"
JAR="/tmp/cityscapes.cookies"
LOGIN_URL="https://www.cityscapes-dataset.com/login/"
GT_URL='https://www.cityscapes-dataset.com/file-handling/?packageID=1'
FOG_URL='https://www.cityscapes-dataset.com/file-handling/?packageID=29'

CS_USER="${CS_USER:-}"
CS_PASS="${CS_PASS:-}"
if [[ -z "$CS_USER" || -z "$CS_PASS" ]]; then
  echo "[ERROR] Set CS_USER and CS_PASS env vars." >&2
  exit 1
fi

OUT="$(cd "$(dirname "$0")/.." && pwd)/datasets/_zips"
mkdir -p "$OUT"

echo "[INFO] Logging in as $CS_USER"
curl -sSL -A "$UA" -c "$JAR" -b "$JAR" "$LOGIN_URL" -o /dev/null
curl -sSL -A "$UA" -c "$JAR" -b "$JAR" -e "$LOGIN_URL" \
  -d "username=$CS_USER" -d "password=$CS_PASS" -d 'submit=Login' \
  "$LOGIN_URL" -o /dev/null

if command -v aria2c >/dev/null 2>&1; then
  echo "[INFO] Downloading with aria2 (parallel)."
  aria2c -x 8 -s 8 -j 2 --continue=true \
    --user-agent="$UA" --load-cookies="$JAR" \
    --dir="$OUT" \
    --out=gtFine_trainvaltest.zip "$GT_URL" \
    --out=leftImg8bit_trainvaltest_foggy.zip "$FOG_URL"
else
  echo "[INFO] aria2 not found; using curl (sequential)."
  curl -fL -A "$UA" -c "$JAR" -b "$JAR" "$GT_URL" -o "$OUT/gtFine_trainvaltest.zip"
  curl -fL -A "$UA" -c "$JAR" -b "$JAR" "$FOG_URL" -o "$OUT/leftImg8bit_trainvaltest_foggy.zip"
fi

echo "[OK] Files in $OUT:" && ls -lh "$OUT"

