#!/usr/bin/env bash
# Launch the YOLO26 x COCO x noise experiment.
#
#   ./run.sh smoke      # coco8, 1 epoch, model n, noise 0/50/95        (minutes)
#   ./run.sh prepare    # download chosen dataset + build on-disk noisy copies, then exit
#   ./run.sh coco8      # coco8   (8 imgs),  5 sizes, all 20 noise levels
#   ./run.sh coco128    # coco128 (128 imgs), 5 sizes, all 20 noise levels
#   ./run.sh reduced    # full COCO, 5 sizes, noise 0/25/50/75/95
#   ./run.sh full       # full COCO, 5 sizes, all 20 noise levels (0-95 step 5)
#   ./run.sh full-xeval # full sweep + the cross-noise validation matrix
#   ./run.sh eval       # ONLY the cross-noise validation matrix, on already-trained weights
#
# Training: each of the 5 scales is trained on every training noise level.
# Cross-evaluation: each trained model (scale x training noise) is then validated across
# the whole validation noise sweep, i.e. sizes x train_levels x val_levels runs.
#
# Env toggles:
#   DATASET=coco128        which dataset (default coco128)
#   NOISY=1                train/eval on the pre-built on-disk noisy datasets
#   BATCH=128              training batch size (default 16). Very large values can exceed
#                          24 GB on the bigger scales; use BATCH=-1 for auto-batch.
#   WORKERS=16             dataloader workers (default 8)
#   OUTPUT_DIR=dir         output directory (default per mode)
#   MODELS_DIR=dir         where trained weights live (…/<name>_noise_<level>/weights/best.pt);
#                          set this for `eval` if the weights are not under <OUTPUT_DIR>/models
#   TRAIN_MIN/TRAIN_MAX/TRAIN_STEP   which trained models to evaluate (default 0/95/5)
#   VAL_MIN/VAL_MAX/VAL_STEP         validation noise sweep per model     (default 0/95/5)
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
    "$PY" noise_experiment.py --smoke $EXTRA --output-dir "${OUTPUT_DIR:-noise_experiment_coco_smoke}" ;;
  prepare)
    "$PY" noise_experiment.py --dataset "$DATASET" --prepare-data \
        --min-noise 0 --max-noise 95 --noise-step 5 --data-root datasets ;;
  coco8)
    "$PY" noise_experiment.py --dataset coco8 --models n s m l x \
        --min-noise 0 --max-noise 95 --noise-step 5 \
        --val-min-noise 0 --val-max-noise 95 --val-noise-step 5 --cross-eval \
        --epochs 100 --batch-size ${BATCH:-16} --workers ${WORKERS:-8} --image-size 640 $EXTRA \
        --output-dir "${OUTPUT_DIR:-noise_experiment_coco8}" ;;
  coco128)
    "$PY" noise_experiment.py --dataset coco128 --models n s m l x \
        --min-noise 0 --max-noise 95 --noise-step 5 \
        --val-min-noise 0 --val-max-noise 95 --val-noise-step 5 --cross-eval \
        --epochs 100 --batch-size ${BATCH:-16} --workers ${WORKERS:-8} --image-size 640 $EXTRA \
        --output-dir "${OUTPUT_DIR:-noise_experiment_coco128}" ;;
  reduced)
    "$PY" noise_experiment.py --dataset coco --models n s m l x --noise-list 0 25 50 75 95 \
        --epochs 100 --batch-size ${BATCH:-16} --workers ${WORKERS:-8} --image-size 640 $EXTRA \
        --output-dir "${OUTPUT_DIR:-noise_experiment_coco}" ;;
  full)
    "$PY" noise_experiment.py --dataset coco --models n s m l x \
        --min-noise 0 --max-noise 95 --noise-step 5 \
        --epochs 100 --batch-size ${BATCH:-16} --workers ${WORKERS:-8} --image-size 640 $EXTRA \
        --output-dir "${OUTPUT_DIR:-noise_experiment_coco}" ;;
  full-xeval)
    "$PY" noise_experiment.py --dataset coco --models n s m l x \
        --min-noise 0 --max-noise 95 --noise-step 5 \
        --epochs 100 --batch-size ${BATCH:-16} --workers ${WORKERS:-8} --image-size 640 --cross-eval $EXTRA \
        --output-dir "${OUTPUT_DIR:-noise_experiment_coco}" ;;
  eval)
    "$PY" noise_experiment.py --dataset "$DATASET" --models n s m l x --eval-only --cross-eval \
        --min-noise "${TRAIN_MIN:-0}" --max-noise "${TRAIN_MAX:-95}" --noise-step "${TRAIN_STEP:-5}" \
        --val-min-noise "${VAL_MIN:-0}" --val-max-noise "${VAL_MAX:-95}" --val-noise-step "${VAL_STEP:-5}" \
        --image-size 640 ${MODELS_DIR:+--models-dir "$MODELS_DIR"} $EXTRA \
        --output-dir "${OUTPUT_DIR:-noise_experiment_coco128}" ;;
  *)
    echo "unknown mode: $MODE (use smoke|prepare|coco8|coco128|reduced|full|full-xeval|eval)"; exit 1 ;;
esac
