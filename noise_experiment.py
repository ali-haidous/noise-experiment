"""
Noise-robustness experiment on COCO with the full YOLO26 model family.

It sweeps image noise across the full YOLO26 model family on COCO: for each noise
level it corrupts the input images by replacing a percentage of pixels with uniform
random values, trains a model, then optionally cross-evaluates every model against
every noise level. It runs across all five YOLO26 scales (n/s/m/l/x) and captures a
memory footprint for every run.

Design notes:

1.  Dataset = COCO (``coco.yaml``). Ultralytics auto-downloads train2017 /
    val2017 (~20 GB) on first use if the dataset is not already present, which
    satisfies "download the dataset if it doesn't exist".

2.  Noise is injected ON-THE-FLY at the network input rather than by writing 20
    physical noisy copies of every image to disk. Twenty noisy copies of COCO's
    118k training images would be ~360 GB and hours of preprocessing. Instead we
    monkeypatch the detection trainer/validator preprocess step and corrupt the
    normalized image tensor there. The corruption is mathematically the same as
    a per-pixel salt-style corruption (replace p%% of pixel positions, across all
    channels, with uniform random values) and is seeded per run for
    reproducibility. This is a deliberate design choice versus staging a
    fixed noisy dataset on disk and is documented in the README.

3.  All five YOLO26 scales are trained: yolo26n / s / m / l / x. Each weight file
    is downloaded automatically on first use.

4.  Memory footprint is captured for every (model, noise) run: peak GPU memory
    (allocated + reserved via the CUDA caching allocator), peak system RSS,
    parameter count, GFLOPs, on-disk weight size, and (in cross-eval) inference
    speed. Results go to ``memory_footprint.csv``.

Run ``python noise_experiment.py --help`` for options. See
README.md for recommended full-run and smoke-test commands.
"""

import argparse
import csv
import gc
import logging
import os
import platform
import threading
import time
from datetime import datetime

try:
    import resource  # Unix-only; optional RSS fallback when psutil is unavailable
except ImportError:  # Windows
    resource = None

import numpy as np
import torch

# ----------------------------------------------------------------------------
# Model family
# ----------------------------------------------------------------------------
YOLO26_SIZES = ["n", "s", "m", "l", "x"]


def size_to_weights(size: str) -> str:
    """'n' -> 'yolo26n.pt'."""
    return f"yolo26{size}.pt"


# ----------------------------------------------------------------------------
# Noise injection
# ----------------------------------------------------------------------------
class _NoiseState:
    """Module-level holder for the currently active noise level. The patched
    preprocess hooks read it, so the same class methods serve every run."""

    def __init__(self):
        self.percentage = 0.0
        self.generator = None
        self.enabled = False

    def configure(self, percentage, seed, device):
        self.percentage = float(percentage)
        self.enabled = self.percentage > 0.0
        gen_device = "cpu" if device.type == "cpu" else device
        self.generator = torch.Generator(device=gen_device)
        self.generator.manual_seed(int(seed))


NOISE = _NoiseState()


def inject_noise_(img):
    """In-place: replace NOISE.percentage%% of pixel positions with uniform random
    values, independently per channel, matching a per-pixel salt-noise corruption.

    ``img`` is a normalized float tensor (B, C, H, W) in [0, 1] on device. A pixel
    position (b, h, w) is selected with probability p; every channel at that
    position is then overwritten with its own uniform random value (spatial mask
    broadcast across channels, independent random value per channel)."""
    if not NOISE.enabled or NOISE.percentage <= 0:
        return img
    p = NOISE.percentage / 100.0
    b, c, h, w = img.shape
    gen = NOISE.generator
    mask = torch.rand((b, 1, h, w), generator=gen, device=img.device) < p
    mask = mask.expand(b, c, h, w)
    rand_vals = torch.randint(
        0, 256, (b, c, h, w), generator=gen, device=img.device, dtype=torch.int16
    ).to(img.dtype) / 255.0
    img[mask] = rand_vals[mask]
    return img


_PATCHED = False


def install_noise_hooks():
    """Monkeypatch detection trainer + validator preprocess to inject noise.

    Both hooks return a batch whose 'img' is a normalized float tensor on device;
    we corrupt it after normalization. Patching the class methods (rather than a
    callback) guarantees noise is applied for training, the automatic end-of-
    training validation, and any standalone model.val() call."""
    global _PATCHED
    if _PATCHED:
        return
    from ultralytics.models.yolo.detect.train import DetectionTrainer
    from ultralytics.models.yolo.detect.val import DetectionValidator

    _orig_train_pre = DetectionTrainer.preprocess_batch
    _orig_val_pre = DetectionValidator.preprocess

    def train_pre(self, batch):
        batch = _orig_train_pre(self, batch)
        inject_noise_(batch["img"])
        return batch

    def val_pre(self, batch):
        batch = _orig_val_pre(self, batch)
        inject_noise_(batch["img"])
        return batch

    DetectionTrainer.preprocess_batch = train_pre
    DetectionValidator.preprocess = val_pre
    _PATCHED = True
    logging.info("Noise injection hooks installed on DetectionTrainer/DetectionValidator.")


# ----------------------------------------------------------------------------
# Memory footprint capture
# ----------------------------------------------------------------------------
def _rss_bytes():
    """Current resident set size of this process, in bytes."""
    try:
        import psutil
        return psutil.Process().memory_info().rss
    except Exception:
        if resource is not None:
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return peak * 1024 if platform.system() != "Darwin" else peak
        return 0


class MemoryMonitor:
    """Context manager capturing peak GPU and system memory + wall time.

    GPU peaks come from the CUDA caching allocator (reset on entry). System RSS is
    sampled on a background thread so the peak reflects this run only."""

    def __init__(self, device, sample_interval=0.5):
        self.device = device
        self.sample_interval = sample_interval
        self._stop = threading.Event()
        self._thread = None
        self.peak_rss = 0
        self.start_time = None
        self.wall_time = 0.0
        self.peak_gpu_alloc = 0
        self.peak_gpu_reserved = 0

    def _sample(self):
        while not self._stop.is_set():
            self.peak_rss = max(self.peak_rss, _rss_bytes())
            self._stop.wait(self.sample_interval)

    def __enter__(self):
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.empty_cache()
        self.peak_rss = _rss_bytes()
        self.start_time = time.time()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self.wall_time = time.time() - self.start_time
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2 * self.sample_interval + 1)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            self.peak_gpu_alloc = torch.cuda.max_memory_allocated(self.device)
            self.peak_gpu_reserved = torch.cuda.max_memory_reserved(self.device)
        # Use the per-run sampled RSS peak. ru_maxrss is a process-wide high-water
        # mark that never resets, so it would over-attribute memory to later runs
        # in a long sweep; only fall back to it if sampling captured nothing.
        if self.peak_rss <= 0 and resource is not None:
            ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            self.peak_rss = ru * 1024 if platform.system() != "Darwin" else ru
        return False

    @property
    def stats(self):
        gb = 1024 ** 3
        return {
            "wall_time_s": round(self.wall_time, 2),
            "peak_gpu_alloc_gb": round(self.peak_gpu_alloc / gb, 3),
            "peak_gpu_reserved_gb": round(self.peak_gpu_reserved / gb, 3),
            "peak_cpu_rss_gb": round(self.peak_rss / gb, 3),
        }


def model_footprint(yolo_model, weights_path, imgsz):
    """Static footprint of a model: params (M), GFLOPs, on-disk size (MB)."""
    fp = {"params_m": None, "gflops": None, "weights_mb": None}
    try:
        from ultralytics.utils.torch_utils import get_flops, get_num_params
        net = yolo_model.model
        fp["params_m"] = round(get_num_params(net) / 1e6, 3)
        fp["gflops"] = round(float(get_flops(net, imgsz)), 2)
    except Exception as e:
        logging.warning(f"footprint helpers failed ({e}); falling back to param count.")
    if fp["params_m"] is None:
        try:
            n = sum(p.numel() for p in yolo_model.model.parameters())
            fp["params_m"] = round(n / 1e6, 3)
        except Exception:
            pass
    try:
        if weights_path and os.path.exists(weights_path):
            fp["weights_mb"] = round(os.path.getsize(weights_path) / 1024 ** 2, 2)
    except Exception:
        pass
    return fp


# ----------------------------------------------------------------------------
# Experiment driver
# ----------------------------------------------------------------------------
class NoiseExperimentCOCO:

    RESULTS_FIELDS = ["model", "size", "noise_level", "epochs", "device",
                      "map50", "map50_95", "precision", "recall"]
    FOOTPRINT_FIELDS = ["model", "size", "noise_level", "phase", "epochs", "imgsz",
                        "batch", "device", "params_m", "gflops", "weights_mb",
                        "wall_time_s", "peak_gpu_alloc_gb", "peak_gpu_reserved_gb",
                        "peak_cpu_rss_gb", "map50", "map50_95", "precision", "recall", "error"]
    EVAL_FIELDS = ["model", "size", "model_noise", "val_noise", "map50", "map50_95",
                   "precision", "recall", "inference_ms_per_img",
                   "peak_gpu_alloc_gb", "peak_cpu_rss_gb"]

    def __init__(self, args):
        self.args = args
        self.data = args.data
        self.sizes = args.models
        if getattr(args, "noise_list", None):
            self.noise_levels = args.noise_list
        else:
            self.noise_levels = list(range(args.min_noise, args.max_noise + args.noise_step, args.noise_step))
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.image_size = args.image_size
        self.output_dir = args.output_dir
        self.seed = args.seed
        self.cross_eval = args.cross_eval
        os.makedirs(self.output_dir, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(os.path.join(self.output_dir, "noise_experiment_coco.log")),
                logging.StreamHandler(),
            ],
        )

        if torch.cuda.is_available() and args.device != "cpu":
            self.device = torch.device("cuda:0")
            props = torch.cuda.get_device_properties(0)
            self.gpu_name = props.name
            logging.info(f"Using GPU: {props.name}")
            logging.info(f"GPU Memory: {props.total_memory / 1024**3:.2f} GB")
            logging.info(f"CUDA Version: {torch.version.cuda}")
        else:
            self.device = torch.device("cpu")
            self.gpu_name = "cpu"
            logging.info(f"Using CPU: {platform.processor() or platform.machine()}")

        self.results_csv = os.path.join(self.output_dir, "training_results.csv")
        self.footprint_csv = os.path.join(self.output_dir, "memory_footprint.csv")
        self.eval_csv = os.path.join(self.output_dir, "evaluation_results.csv")

    def _append_row(self, path, row, fieldnames):
        exists = os.path.exists(path)
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, restval="", extrasaction="ignore")
            if not exists:
                w.writeheader()
            w.writerow(row)

    def _already_done(self, path, key_fields, key_values):
        if not os.path.exists(path):
            return False
        try:
            with open(path) as f:
                for r in csv.DictReader(f):
                    if all(str(r.get(k)) == str(v) for k, v in zip(key_fields, key_values)):
                        return True
        except Exception:
            return False
        return False

    def _train_args(self, size, noise_level):
        run_name = f"yolo26{size}_noise_{noise_level}"
        ta = {
            "data": self.data,
            "epochs": self.epochs,
            "imgsz": self.image_size,
            "batch": self.batch_size,
            "device": 0 if self.device.type == "cuda" else "cpu",
            "name": run_name,
            "project": os.path.join(self.output_dir, "models"),
            "patience": self.args.patience,
            "optimizer": "auto",
            "verbose": True,
            "exist_ok": True,
            "pretrained": True,
            "close_mosaic": 10,
            "seed": self.seed,
            "val": True,
            "cos_lr": True,
        }
        if self.device.type == "cpu":
            ta.update({"batch": min(8, self.batch_size), "amp": False,
                       "workers": max(1, (os.cpu_count() or 2) // 2), "plots": False})
        else:
            ta.update({"amp": True, "workers": self.args.workers})
        return ta, run_name

    def train_all(self):
        from ultralytics import YOLO

        total = len(self.sizes) * len(self.noise_levels)
        idx = 0
        for size in self.sizes:
            weights = size_to_weights(size)
            for noise_level in self.noise_levels:
                idx += 1
                banner = f"[{idx}/{total}] model=yolo26{size} noise={noise_level}%"
                if self._already_done(self.footprint_csv, ["model", "noise_level", "phase"],
                                      [f"yolo26{size}", noise_level, "train"]):
                    logging.info(f"{banner} -- already recorded, skipping.")
                    continue
                logging.info(f"=== {banner} ===")

                # Configure noise for this run BEFORE building the trainer.
                NOISE.configure(noise_level, seed=self.seed + noise_level, device=self.device)

                try:
                    model = YOLO(weights)   # auto-downloads the weight file on first use
                    fp = model_footprint(model, self._weights_file(weights), self.image_size)
                    train_args, run_name = self._train_args(size, noise_level)

                    with MemoryMonitor(self.device, self.args.mem_interval) as mem:
                        results = model.train(**train_args)

                    metrics = self._extract_metrics(results)
                    best_pt = os.path.join(self.output_dir, "models", run_name, "weights", "best.pt")
                    trained_fp = model_footprint(model, best_pt, self.image_size)

                    row = {
                        "model": f"yolo26{size}", "size": size, "noise_level": noise_level,
                        "phase": "train", "epochs": self.epochs, "imgsz": self.image_size,
                        "batch": train_args["batch"], "device": self.gpu_name,
                        "params_m": fp["params_m"], "gflops": fp["gflops"],
                        "weights_mb": trained_fp["weights_mb"] or fp["weights_mb"],
                        **mem.stats, **metrics,
                    }
                    self._append_row(self.results_csv, row, self.RESULTS_FIELDS)
                    self._append_row(self.footprint_csv, row, self.FOOTPRINT_FIELDS)
                    logging.info(f"{banner} done: mAP50-95={metrics['map50_95']}, "
                                 f"peak_gpu_alloc={mem.stats['peak_gpu_alloc_gb']} GB, "
                                 f"peak_cpu_rss={mem.stats['peak_cpu_rss_gb']} GB, "
                                 f"time={mem.stats['wall_time_s']}s")
                except Exception as e:
                    logging.error(f"{banner} FAILED: {e}", exc_info=True)
                    self._append_row(self.footprint_csv, {
                        "model": f"yolo26{size}", "size": size, "noise_level": noise_level,
                        "phase": "train", "error": str(e)}, self.FOOTPRINT_FIELDS)
                finally:
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()
                    gc.collect()

    def cross_evaluate(self):
        """For every trained model, validate against every noise level (sizes x N x N)."""
        from ultralytics import YOLO
        logging.info("Starting cross-noise evaluation (the expensive N x N matrix).")
        for size in self.sizes:
            for model_noise in self.noise_levels:
                run_name = f"yolo26{size}_noise_{model_noise}"
                best_pt = os.path.join(self.output_dir, "models", run_name, "weights", "best.pt")
                if not os.path.exists(best_pt):
                    logging.warning(f"missing weights for {run_name}, skipping.")
                    continue
                for val_noise in self.noise_levels:
                    if self._already_done(self.eval_csv, ["model", "model_noise", "val_noise"],
                                          [f"yolo26{size}", model_noise, val_noise]):
                        continue
                    NOISE.configure(val_noise, seed=self.seed + val_noise, device=self.device)
                    try:
                        model = YOLO(best_pt)
                        with MemoryMonitor(self.device, self.args.mem_interval) as mem:
                            res = model.val(data=self.data,
                                            device=0 if self.device.type == "cuda" else "cpu",
                                            name=f"{run_name}_val_{val_noise}",
                                            project=os.path.join(self.output_dir, "runs"),
                                            exist_ok=True, verbose=False)
                        m = self._extract_metrics(res)
                        speed = getattr(res, "speed", {}) or {}
                        self._append_row(self.eval_csv, {
                            "model": f"yolo26{size}", "size": size,
                            "model_noise": model_noise, "val_noise": val_noise, **m,
                            "inference_ms_per_img": round(speed.get("inference", float("nan")), 3),
                            "peak_gpu_alloc_gb": mem.stats["peak_gpu_alloc_gb"],
                            "peak_cpu_rss_gb": mem.stats["peak_cpu_rss_gb"],
                        }, self.EVAL_FIELDS)
                    except Exception as e:
                        logging.error(f"val {run_name} @ {val_noise}% FAILED: {e}")
                    finally:
                        if self.device.type == "cuda":
                            torch.cuda.empty_cache()
                        gc.collect()

    @staticmethod
    def _extract_metrics(results):
        try:
            return {
                "map50": round(float(results.box.map50), 5),
                "map50_95": round(float(results.box.map), 5),
                "precision": round(float(results.box.mp), 5),
                "recall": round(float(results.box.mr), 5),
            }
        except Exception:
            return {"map50": None, "map50_95": None, "precision": None, "recall": None}

    @staticmethod
    def _weights_file(weights):
        return weights

    def run(self):
        start = datetime.now()
        logging.info(f"Noise x YOLO26 x COCO experiment start: sizes={self.sizes}, "
                     f"noise={self.noise_levels}, data={self.data}, epochs={self.epochs}")
        install_noise_hooks()
        self.train_all()
        if self.cross_eval:
            self.cross_evaluate()
        logging.info(f"Experiment complete in {datetime.now() - start}. "
                     f"Footprint -> {self.footprint_csv}")


def parse_args():
    p = argparse.ArgumentParser(description="Noise robustness across YOLO26 sizes on COCO")
    p.add_argument("--dataset", choices=["coco", "coco8", "coco128"], default="coco",
                   help="Which dataset to train on, auto-downloaded if missing: "
                        "coco = full train2017/val2017 (~20 GB), "
                        "coco128 = 128-image subset, coco8 = 8-image subset.")
    p.add_argument("--data", type=str, default=None,
                   help="Raw Ultralytics dataset yaml override. If omitted it is "
                        "derived from --dataset (e.g. coco128 -> coco128.yaml).")
    p.add_argument("--models", nargs="+", default=YOLO26_SIZES, choices=YOLO26_SIZES,
                   help="YOLO26 scales to run (default: all five).")
    p.add_argument("--image-size", type=int, default=640)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--min-noise", type=int, default=0)
    p.add_argument("--max-noise", type=int, default=95)
    p.add_argument("--noise-step", type=int, default=5)
    p.add_argument("--noise-list", type=int, nargs="+", default=None,
                   help="Explicit noise levels (overrides min/max/step), e.g. --noise-list 0 25 50 75 95")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto", help="'auto', 'cpu', or a CUDA index like '0'")
    p.add_argument("--output-dir", type=str, default="noise_experiment_coco")
    p.add_argument("--cross-eval", action="store_true",
                   help="After training, validate every model against every noise level "
                        "(sizes x N x N -- very expensive on full COCO).")
    p.add_argument("--mem-interval", type=float, default=0.5, help="RSS sampling interval (s)")
    p.add_argument("--smoke", action="store_true",
                   help="Smoke test: coco8.yaml, 1 epoch, imgsz 320, noise levels 0/50/95, model n only.")
    args = p.parse_args()

    if args.smoke:
        args.dataset = "coco8"
        args.data = None
        args.epochs = 1
        args.image_size = 320
        args.batch_size = 4
        args.models = ["n"]
        args.noise_list = [0, 50, 95]
    # Resolve the friendly --dataset switch into a yaml unless --data overrides it.
    if args.data is None:
        args.data = f"{args.dataset}.yaml"
    if args.noise_list and (min(args.noise_list) < 0 or max(args.noise_list) > 100):
        p.error("Noise levels must be between 0 and 100")
    if args.min_noise < 0 or args.max_noise > 100:
        p.error("Noise levels must be between 0 and 100")
    if args.min_noise > args.max_noise:
        p.error("min_noise must be <= max_noise")
    if args.noise_step <= 0:
        p.error("noise_step must be positive")
    return args


if __name__ == "__main__":
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    NoiseExperimentCOCO(args).run()
