# Noise experiment — COCO × YOLO26 (all sizes) with memory-footprint capture

This experiment sweeps image noise across the full YOLO26 model family on COCO: for each
noise level it corrupts the input images with a percentage of uniform-random pixels,
trains a model, and optionally cross-evaluates every model against every noise level —
recording accuracy and a memory footprint per run. It:

- trains **all five YOLO26 scales** (`yolo26n/s/m/l/x`, auto-downloaded);
- trains on **COCO** (`train2017`/`val2017`), auto-downloaded if not already present;
- **captures a memory footprint** for every run.

Main script: `noise_experiment.py`.

---

## Install

The RTX 4090/5090 needs a CUDA build of PyTorch (same as the original setup, CUDA 13.2):

First, install CUDA 13.2: https://developer.download.nvidia.com/compute/cuda/13.2.0/local_installers/cuda_13.2.0_windows.exe
Then, run:
```bash
chmod +x ./create_env.sh
./create_env.sh
source .venv/Scripts/activate     # (.venv/bin/activate on Linux/mac)
```

`ultralytics>=8.4.90` is required — that is the first series shipping YOLO26.

---

## Run

Convenience wrapper (`run.sh`):

```bash
./run.sh smoke        # coco8, 1 epoch, model n, noise 0/50/95  — minutes, sanity check
./run.sh coco8        # coco8 (8 imgs),   5 sizes, all 20 noise levels — fast full-shape run
./run.sh subset       # first 512 COCO images, 5 sizes, all 20 noise levels (NUM_IMAGES=1024 ./run.sh subset for more)
./run.sh reduced      # full COCO, 5 sizes, noise 0/25/50/75/95 (25 trainings)
./run.sh full         # full COCO, 5 sizes, all 20 noise levels  (100 trainings)
./run.sh full-xeval   # full sweep + sizes × N × N cross-evaluation matrix
```

Or call the script directly:

```bash
# full sweep: 5 sizes × 20 noise levels (0–95 step 5)
python noise_experiment.py \
    --dataset coco --models n s m l x \
    --min-noise 0 --max-noise 95 --noise-step 5 \
    --epochs 100 --batch-size 16 --image-size 640 \
    --output-dir noise_experiment_coco
```

Useful flags: `--models n s` (subset of scales), `--noise-list 0 25 50 75 95`
(explicit levels, overrides min/max/step), `--epochs`, `--batch-size`,
`--device 0|cpu`, `--cross-eval`, `--smoke`.

**Dataset switch:** `--dataset {coco,coco8}` (default `coco`) picks what to train on —
`coco` is the full train2017/val2017 (~20 GB) and `coco8` an 8-image smoke-test set. Both
auto-download if missing. `--data <yaml>` remains a raw override that wins over `--dataset`
if you want a custom dataset yaml.

### COCO subsets: the first N images

`--num-images N` trains on just the first N images of COCO train2017 (sorted by file name,
the same ordering coco128 uses, so `--num-images 128` trains on coco128's images). Pick any size,
for example 512, 1024 or 2048:

```bash
NUM_IMAGES=1024 ./run.sh subset                     # or call the script directly:
python noise_experiment.py --dataset coco --num-images 2048 --models n --noise-list 0 50 95
```

How it works:

- Full COCO 2017 (labels + train2017 + val2017, ~20 GB) is downloaded **once** into
  `datasets/coco/`. Anything already there is reused, the 7 GB test2017 split is skipped,
  and each zip is deleted after extraction.
- The subset is built at `datasets/coco_first<N>/` with its own `train.txt`, `val.txt` and
  `coco_first<N>.yaml`. Images are hard-linked from `datasets/coco` (no extra disk; copied if
  linking isn't possible) and labels are copied, so it has its own label cache. Later runs
  with the same N reuse it; `--force-regenerate` rebuilds it.
- Validation uses the first N images of **val2017** (held out, unlike coco128 which validates
  on its training images). Set `--val-images M` (`VAL_IMAGES=M` in `run.sh`, max 5000) to
  validate on a different count, e.g. `--val-images 5000` for the full val2017.
- Each subset is a normal dataset to the rest of the pipeline, so `NOISY=1`, `prepare`,
  `--cross-eval` and `eval` all work on it. Results default to
  `noise_experiment_coco_first<N>/`.

### On-disk noisy datasets

By default the noise is injected **on the fly** during training and nothing extra is
written to disk. If you'd rather materialize the corrupted images — to inspect them, or to
train on a fixed noisy set — build them once and train on them:

```bash
./run.sh prepare                 # download + build noisy copies for the first 512 COCO images, then exit
NUM_IMAGES=2048 ./run.sh prepare # ...a bigger subset
DATASET=coco8 ./run.sh prepare   # ...or coco8 / coco
NOISY=1 ./run.sh subset          # train on the pre-built on-disk noisy datasets
```

Both the clean dataset and its noisy copies are written under `./datasets/` (override with
`--data-root`):

```
datasets/
  coco/                         # full COCO 2017 (downloaded once)
  coco_first512/                # first-512 subset built from it
  coco_first512_noisy/
    noise_0/   { images/  labels/  train.txt  val.txt  dataset.yaml }
    noise_5/   ...
    ...
```

Each `noise_<level>/` is a self-contained YOLO dataset — a fixed, reproducible corruption
of the clean images (labels copied unchanged), so runs are repeatable and the images are
inspectable. Direct flags: `--prepare-data` (build then exit), `--use-noisy-data`
(train/eval on the on-disk copies, auto-building if missing), `--force-regenerate`
(rebuild), `--data-root DIR`. Materializing all 20 levels of **full COCO** is ~360 GB, so
prefer a `--num-images` subset or coco8 for the on-disk path, or fewer noise levels.

### Cross-noise validation matrix

Training produces one model per (scale, training noise level). The cross-evaluation step then
validates each of those models across a full validation noise sweep, so you can see how a model
trained at one noise level holds up when tested at every other level. The matrix has
`sizes x train_levels x val_levels` cells (for the default sweep that is 5 x 20 x 20 = 2000
validations).

```bash
./run.sh full-xeval                    # matrix right after training (or add --cross-eval)
MODELS_DIR=path/to/models ./run.sh eval # ONLY the matrix, on weights you already trained
```

The validation sweep is configured independently of the training levels:

- `--val-min-noise`, `--val-max-noise`, `--val-noise-step`, or `--val-noise-list` set the levels
  each model is validated on. If you omit them, the validation sweep matches the training levels.
- `--cross-eval` runs the matrix after training; `--eval-only` runs just the matrix on existing
  weights (and implies `--cross-eval`).
- `--models-dir DIR` points evaluation at the trained weights
  (`DIR/<name>_noise_<level>/weights/best.pt`), defaulting to `<output-dir>/models`. Ultralytics
  can save runs under its own `runs` folder, so if `eval` reports missing weights, set
  `--models-dir` (or `MODELS_DIR`) to the path shown next to "Results saved to" in the training log.

Results go to `evaluation_results.csv`, one row per cell: `model, size, model_noise, val_noise,
mAP50, mAP50_95, precision, recall`, the validation speed per image (`preprocess_ms_per_img`,
`inference_ms_per_img`, `postprocess_ms_per_img`), and peak GPU and host memory. The speed columns
are the per-image latency Ultralytics reports at that noise level, so each cell shows both how
accuracy and how inference cost respond as the validation noise changes.
Re-running resumes and skips cells already recorded.

```bash
# validate every trained model on the full 0 to 95 sweep, step 5
python noise_experiment.py --dataset coco --num-images 512 --eval-only --cross-eval \
    --min-noise 0 --max-noise 95 --noise-step 5 \
    --val-min-noise 0 --val-max-noise 95 --val-noise-step 5 \
    --models-dir noise_experiment_coco_first512/models --output-dir noise_experiment_coco_first512
```

**The COCO download (~20 GB) happens automatically** the first time you use `--dataset coco`
(with or without `--num-images`) and lands in `./datasets/coco/` (or `--data-root`). No manual
download needed; if COCO is already present it is reused. `coco8` downloads the same way
(a few MB).

---

## The noise, and the one deliberate deviation

A naive approach would write 20 physically-noised copies of the dataset to disk. For COCO that
would be ~360 GB (20 × 118k images) and hours of preprocessing, so instead the noise
is injected **on-the-fly at the network input**: the detection trainer/validator
`preprocess` step is monkeypatched, and right after images are normalized to `[0,1]`
we replace `p%` of pixel positions (across all channels, independent random value per
channel) with uniform random values. This is mathematically the same corruption as a
reference salt-noise routine (verified: measured corruption fractions match the target within
1% — e.g. 5%→0.047, 50%→0.508, 95%→0.950), on the standard `0..255/255` value grid.

Consequences to be aware of:

- Noise is applied **after** augmentation (mosaic/scale/flip), i.e. at the true model
  input, and regenerated each batch. It is seeded per run (`seed + noise_level`) for
  reproducibility. This behaves like a fixed-intensity stochastic noise rather than one
  frozen noisy image set — the appropriate analog when you can't stage 360 GB on disk.
- `noise=0` is a true no-op (clean baseline). `noise=100` fully randomizes every pixel.

If you specifically need the *frozen-on-disk* variant for a subset of levels, say so and
it can be added as an offline pre-pass.

---

This on-the-fly path is the default. You can also materialize the noisy images on disk and train on them instead — see **On-disk noisy datasets** above.

## Memory footprint outputs

Everything is written under `--output-dir` (default `noise_experiment_coco/`):

- **`memory_footprint.csv`** — one row per training run:
  `model, size, noise_level, phase, epochs, imgsz, batch, device, params_m, gflops,
  weights_mb, wall_time_s, peak_gpu_alloc_gb, peak_gpu_reserved_gb, peak_cpu_rss_gb,
  map50, map50_95, precision, recall, error`
- **`training_results.csv`** — compact accuracy table (mAP/precision/recall per run).
- **`evaluation_results.csv`** — only with `--cross-eval`: per (model_noise, val_noise) mAP, the
  validation speed per image (`preprocess_ms_per_img`, `inference_ms_per_img`,
  `postprocess_ms_per_img`), and peak memory.
- **`noise_experiment_coco.log`** — full run log (device banner, per-run summary).
- **`models/yolo26<size>_noise_<level>/`** — standard Ultralytics run dir (weights, plots).

How the numbers are measured: GPU peaks come from the CUDA caching allocator
(`reset_peak_memory_stats` on entry, `max_memory_allocated` / `max_memory_reserved` on
exit — so each run is measured independently). System RSS is sampled on a background
thread over the run window (per-run, not the process-wide high-water mark). Params/GFLOPs
come from Ultralytics' own helpers at the configured `imgsz`. Reference (imgsz 640):

| scale | params (M) | GFLOPs |
|-------|-----------:|-------:|
| n     |       2.57 |    6.1 |
| s     |      10.01 |   22.8 |
| m     |      21.90 |   75.4 |
| l     |      26.30 |   93.8 |
| x     |      58.99 |  209.5 |

---

## Cost / time — please read before launching `full`

A single small model over 877 images (the road-sign predecessor) still took ~18 h–2 days.
COCO has ~118k training images (~135×), and this sweep is **5 sizes × 20 noise levels =
100 full-COCO trainings** at 100 epochs each. On a single RTX 4090 that is realistically
**on the order of months** of wall-clock (roughly ~half a day for `n` up to several days
for `x`, per noise level). Options to make it tractable:

- `reduced` mode (noise 0/25/50/75/95) → 25 trainings, ~4× less.
- Lower `--epochs` (e.g. 50), or fewer `--models`.
- Distribute noise levels / sizes across multiple GPUs or machines.

`--cross-eval` adds a **sizes × 20 × 20 = 2000** COCO validations on top — days more —
so it is off by default. The on-the-fly noise itself adds negligible overhead (one
masked tensor op per batch).

**Resuming is safe:** each completed run is recorded in `memory_footprint.csv`; re-running
the same command skips finished `(model, noise)` runs and continues where it stopped.

---

## What was validated where

Because this sandbox has no GPU (and the CPU PyTorch wheels were network-blocked), the
**full training could not be executed here**. What *was* verified in-sandbox against the
real `ultralytics==8.4.90` source:

- YOLO26 configs and the `n/s/m/l/x` scales exist; `coco.yaml` auto-download is wired.
- The injection points (`DetectionTrainer.preprocess_batch`, `DetectionValidator.preprocess`)
  return the normalized `img` tensor, so the monkeypatch corrupts the true model input for
  both training and validation.
- The noise algorithm reproduces the target corruption fractions exactly and is
  seed-deterministic (unit-tested), and the memory monitor captures peak RSS + wall time.
- The driver (argument parsing, the 100-run plan, CSV schemas that stay aligned even when
  a run fails, and the resume/skip logic) all pass unit tests.

Run `./run.sh smoke` on the GPU box first — it exercises the entire pipeline
end-to-end on coco8 in a couple of minutes before committing to the long sweep.
