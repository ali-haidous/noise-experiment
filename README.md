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

The RTX 4090 needs a CUDA build of PyTorch (same as the original setup, CUDA 12.6):

```bash
python -m venv .venv && source .venv/Scripts/activate     # (.venv/bin/activate on Linux/mac)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

`ultralytics>=8.4.90` is required — that is the first series shipping YOLO26.

---

## Run

Convenience wrapper (`run.sh`):

```bash
./run.sh smoke        # coco8, 1 epoch, model n, noise 0/50/95  — minutes, sanity check
./run.sh coco8        # coco8 (8 imgs),   5 sizes, all 20 noise levels — fast full-shape run
./run.sh coco128      # coco128 (128 imgs), 5 sizes, all 20 noise levels
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

**Dataset switch:** `--dataset {coco,coco8,coco128}` (default `coco`) picks what to train
on — `coco` is the full train2017/val2017 (~20 GB), `coco128` is a 128-image subset, and
`coco8` an 8-image subset. All three auto-download if missing. Use `coco8`/`coco128` to run
the entire experiment shape (all sizes × all noise levels + memory capture) in minutes
before committing to full COCO. `--data <yaml>` remains a raw override that wins over
`--dataset` if you want a custom dataset yaml.

### On-disk noisy datasets

By default the noise is injected **on the fly** during training and nothing extra is
written to disk. If you'd rather materialize the corrupted images — to inspect them, or to
train on a fixed noisy set — build them once and train on them:

```bash
./run.sh prepare                 # download + build noisy copies for coco128, then exit
DATASET=coco8 ./run.sh prepare   # ...or coco8 / coco
NOISY=1 ./run.sh coco128         # train on the pre-built on-disk noisy datasets
```

Both the clean dataset and its noisy copies are written under `./datasets/` (override with
`--data-root`):

```
datasets/
  coco128/                      # clean dataset (auto-downloaded here)
  coco128_noisy/
    noise_0/   { images/  labels/  train.txt  val.txt  dataset.yaml }
    noise_5/   ...
    ...
```

Each `noise_<level>/` is a self-contained YOLO dataset — a fixed, reproducible corruption
of the clean images (labels copied unchanged), so runs are repeatable and the images are
inspectable. Direct flags: `--prepare-data` (build then exit), `--use-noisy-data`
(train/eval on the on-disk copies, auto-building if missing), `--force-regenerate`
(rebuild), `--data-root DIR`. Materializing all 20 levels of **full COCO** is ~360 GB, so
prefer coco8/coco128 for the on-disk path, or fewer noise levels.

**The COCO download (~20 GB) is handled by Ultralytics** on first training run — it
lands in the Ultralytics `datasets/` directory (see `yolo settings` for the path).
No manual download needed; if COCO is already present it is reused. The `coco8` and
`coco128` subsets download the same way (a few MB each).

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
- **`evaluation_results.csv`** — only with `--cross-eval`: per (model_noise, val_noise)
  mAP plus `inference_ms_per_img` and peak memory.
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
