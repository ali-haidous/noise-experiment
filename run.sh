#!/usr/bin/env bash
# Launch the YOLO26 x COCO x noise experiment.
#
#   ./run.sh smoke     # coco8, 1 epoch, model n, noise 0/50/95   (minutes)
#   ./run.sh coco8     # coco8   (8 imgs),  5 sizes, all 20 noise levels
#   ./run.sh coco128   # coco128 (128 imgs), 5 sizes, all 20 noise levels
#   ./run.sh reduced   # full COCO, 5 sizes, noise 0/25/50/75/95
#   ./run.sh full      # full COCO, 5 sizes, all 20 noise levels (0-95 step 5)
#   ./run.sh full-xeval# full sweep + the sizes x N x N cross-evaluation matrix
#
# coco8 / coco128 auto-download (tiny) and run the FULL experiment shape fast --
# handy for validating the whole pipeline + memory capture before the real COCO run.
# Activate your venv first (e.g. `source .venv/Scripts/activate` on Windows bash,
# or `source .venv/bin/activate` on Linux/mac).
set -e
MODE="${1:-smoke}"
PY="${PYTHON:-python}"
case "$MODE" in
  smoke)
    "$PY" noise_experiment.py --smoke --output-dir noise_experiment_coco_smoke ;;
  coco8)
    "$PY" noise_experiment.py \
        --dataset coco8 --models n s m l x --min-noise 0 --max-noise 95 --noise-step 5 \
        --epochs 100 --batch-size 16 --image-size 640 \
        --output-dir noise_experiment_coco8 ;;
  coco128)
    "$PY" noise_experiment.py \
        --dataset coco128 --models n s m l x --min-noise 0 --max-noise 95 --noise-step 5 \
        --epochs 100 --batch-size 16 --image-size 640 \
        --output-dir noise_experiment_coco128 ;;
  reduced)
    "$PY" noise_experiment.py \
        --dataset coco --models n s m l x --noise-list 0 25 50 75 95 \
        --epochs 100 --batch-size 16 --image-size 640 \
        --output-dir noise_experiment_coco ;;
  full)
    "$PY" noise_experiment.py \
        --dataset coco --models n s m l x --min-noise 0 --max-noise 95 --noise-step 5 \
        --epochs 100 --batch-size 16 --image-size 640 \
        --output-dir noise_experiment_coco ;;
  full-xeval)
    "$PY" noise_experiment.py \
        --dataset coco --models n s m l x --min-noise 0 --max-noise 95 --noise-step 5 \
        --epochs 100 --batch-size 16 --image-size 640 --cross-eval \
        --output-dir noise_experiment_coco ;;
  *)
    echo "unknown mode: $MODE (use smoke|coco8|coco128|reduced|full|full-xeval)"; exit 1 ;;
esac
