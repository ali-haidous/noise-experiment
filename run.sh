#!/usr/bin/env bash
# Launch the YOLO26 x COCO x noise experiment.
#
#   ./run.sh smoke      # coco8, 1 epoch, model n, noise 0/50/95        (minutes)
#   ./run.sh prepare    # download chosen dataset + build on-disk noisy copies, then exit
#   ./run.sh coco8      # coco8   (8 imgs),  5 sizes, all 20 noise levels
#   ./run.sh coco128    # coco128 (128 imgs), 5 sizes, all 20 noise levels
#   ./run.sh reduced    # full COCO, 5 sizes, noise 0/25/50/75/95
#   ./run.sh full       # full COCO, 5 sizes, all 20 noise levels (0-95 step 5)
#   ./run.sh full-xeval # full sweep + the sizes x N x N cross-evaluation matrix
#
# Data is materialized under ./datasets/ : the clean dataset (<name>/) and the noisy
# copies (<name>_noisy/noise_<level>/) live together there.
#
# Env toggles:
#   DATASET=coco128   choose which dataset 'prepare' targets (default coco128)
#   NOISY=1           train/eval on the pre-built on-disk noisy datasets instead of
#                     injecting noise on the fly (auto-builds them first if missing)
#
# Activate your venv first (source .venv/Scripts/activate on Windows bash).
set -e
MODE="${1:-smoke}"
PY="${PYTHON:-python}"
DATASET="${DATASET:-coco128}"
EXTRA=""
[ "${NOISY:-0}" = "1" ] && EXTRA="--use-noisy-data"

case "$MODE" in
  smoke)
    "$PY" noise_experiment.py --smoke $EXTRA --output-dir noise_experiment_coco_smoke ;;
  prepare)
    "$PY" noise_experiment.py --dataset "$DATASET" --prepare-data \
        --min-noise 0 --max-noise 95 --noise-step 5 --data-root datasets ;;
  coco8)
    "$PY" noise_experiment.py --dataset coco8 --models n s m l x \
        --min-noise 0 --max-noise 95 --noise-step 5 \
        --epochs 100 --batch-size 16 --image-size 640 $EXTRA \
        --output-dir noise_experiment_coco8 ;;
  coco128)
    "$PY" noise_experiment.py --dataset coco128 --models n s m l x \
        --min-noise 0 --max-noise 95 --noise-step 5 \
        --epochs 100 --batch-size 16 --image-size 640 $EXTRA \
        --output-dir noise_experiment_coco128 ;;
  reduced)
    "$PY" noise_experiment.py --dataset coco --models n s m l x --noise-list 0 25 50 75 95 \
        --epochs 100 --batch-size 16 --image-size 640 $EXTRA \
        --output-dir noise_experiment_coco ;;
  full)
    "$PY" noise_experiment.py --dataset coco --models n s m l x \
        --min-noise 0 --max-noise 95 --noise-step 5 \
        --epochs 100 --batch-size 16 --image-size 640 $EXTRA \
        --output-dir noise_experiment_coco ;;
  full-xeval)
    "$PY" noise_experiment.py --dataset coco --models n s m l x \
        --min-noise 0 --max-noise 95 --noise-step 5 \
        --epochs 100 --batch-size 16 --image-size 640 --cross-eval $EXTRA \
        --output-dir noise_experiment_coco ;;
  *)
    echo "unknown mode: $MODE (use smoke|prepare|coco8|coco128|reduced|full|full-xeval)"; exit 1 ;;
esac
