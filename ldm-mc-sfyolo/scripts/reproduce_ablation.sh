#!/usr/bin/env bash
set -euo pipefail

# Example ablation runs for SF-YOLO on Foggy Cityscapes target-only UDA
# Assumes you have pre-downloaded Foggy Cityscapes and set data to data/foggy_cityscapes.yaml

PY=${PY:-python}
WEIGHTS=${WEIGHTS:-yolov5s.pt}
DATA=${DATA:-data/foggy_cityscapes.yaml}
HYP=${HYP:-data/hyps/hyp.scratch-low.yaml}
EPOCHS=${EPOCHS:-50}
BATCH=${BATCH:-8}
IMGSZ=${IMGSZ:-960}

# 1) LDM only
$PY run_adaptation.py \
  --data $DATA --weights $WEIGHTS --hyp $HYP --epochs $EPOCHS --batch-size $BATCH --imgsz $IMGSZ \
  --project runs/ablation --name ldm_only --exist-ok \
  --ta_method ldm --mc_T 0

# 2) MC only: T=5/10/15
for T in 5 10 15; do
  $PY run_adaptation.py \
    --data $DATA --weights $WEIGHTS --hyp $HYP --epochs $EPOCHS --batch-size $BATCH --imgsz $IMGSZ \
    --project runs/ablation --name mc_only_T${T} --exist-ok \
    --ta_method tam --mc_dropout --mc_T $T --mc_p 0.2
done

# 3) LDM + MC: T=5/10/15
for T in 5 10 15; do
  $PY run_adaptation.py \
    --data $DATA --weights $WEIGHTS --hyp $HYP --epochs $EPOCHS --batch-size $BATCH --imgsz $IMGSZ \
    --project runs/ablation --name ldm_mc_T${T} --exist-ok \
    --ta_method ldm --mc_dropout --mc_T $T --mc_p 0.2
done

echo "Ablation runs finished. Check runs/ablation/* for results."

