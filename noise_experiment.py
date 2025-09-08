import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
from datetime import datetime
import yaml
from ultralytics import YOLO
import torch
from PIL import Image
import glob
from pathlib import Path
import cv2
from typing import Union, Optional
import warnings
import argparse
import xml.etree.ElementTree as ET
import multiprocessing as mp
from multiprocessing import Queue
import platform
import json
from rich.live import Live
from rich.table import Table
from queue import Empty
import time


def get_existing_metrics(results_dir):
    """Extract metrics from an existing YOLO validation run directory."""
    try:
        # Look for the results.json file in the run directory
        results_file = os.path.join(results_dir, "results.json")
        if not os.path.exists(results_file):
            return None
            
        with open(results_file, 'r') as f:
            results = json.load(f)
            
        # Extract relevant metrics from the results
        metrics = {
            'mAP50-95': float(results.get('metrics/mAP50-95', 0)),
            'mAP50': float(results.get('metrics/mAP50', 0)),
            'precision': float(results.get('metrics/precision', 0)),
            'recall': float(results.get('metrics/recall', 0))
        }
        
        return metrics
    except Exception as e:
        logging.warning(f"Failed to parse existing results from {results_dir}: {str(e)}")
        return None


def add_noise(
    input_path_or_image: Union[str, Image.Image, np.ndarray],
    percentage: float,
    output_path: Optional[str] = None,
    seed: Optional[int] = None,
    preserve_channels: bool = True
) -> Optional[Union[Image.Image, np.ndarray]]:
    """
    Adds noise to an image by randomly setting pixel values based on percentage.
    For RGB images, generates random values between 0-255 for each channel.
    For grayscale images, generates random values between 0-255.
    """
    if not isinstance(percentage, (int, float)):
        raise TypeError(f"Percentage must be a number, got {type(percentage)}")
    if not 0 <= percentage <= 100:
        raise ValueError(f"Percentage must be between 0-100, got {percentage}")
        
    if seed is not None:
        np.random.seed(seed)
    
    if isinstance(input_path_or_image, str):
        input_type = "path"
        if not os.path.exists(input_path_or_image):
            raise FileNotFoundError(f"Image file not found: {input_path_or_image}")
            
        img = cv2.imread(input_path_or_image, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Could not open image file: {input_path_or_image}")
            
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
    elif isinstance(input_path_or_image, Image.Image):
        input_type = "pil"
        img = np.array(input_path_or_image)
        
    elif isinstance(input_path_or_image, np.ndarray):
        input_type = "numpy"
        img = input_path_or_image.copy()
        
    else:
        raise TypeError(f"Input must be a file path, PIL Image, or numpy array, got {type(input_path_or_image)}")

    # Create noise mask
    if preserve_channels and len(img.shape) == 3:
        noise_mask = np.random.random(img.shape[:2]) < (percentage / 100)
        noise_mask = np.expand_dims(noise_mask, axis=-1)
        noise_mask = np.broadcast_to(noise_mask, img.shape)
        
        # Generate random RGB values where noise_mask is True
        random_noise = np.random.randint(0, 256, size=img.shape, dtype=np.uint8)
        img = np.where(noise_mask, random_noise, img)
    else:
        noise_mask = np.random.random(img.shape) < (percentage / 100)
        random_noise = np.random.randint(0, 256, size=img.shape, dtype=np.uint8)
        img = np.where(noise_mask, random_noise, img)
    
    if output_path:
        if input_type == "path":
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            success = cv2.imwrite(output_path, img)
            if not success:
                warnings.warn(f"Failed to save image to {output_path}")
            return None
        elif input_type == "pil":
            Image.fromarray(img).save(output_path)
            return None
    
    if input_type == "pil":
        return Image.fromarray(img)
    return img

def convert_xml_to_yolo(xml_path, img_width, img_height):
    """Convert XML annotations to YOLO format."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    class_map = {
        'speedlimit': 0,
        'stop': 1,
        'crosswalk': 2,
        'trafficlight': 3
    }
    
    yolo_annotations = []
    
    for obj in root.findall('object'):
        class_name = obj.find('name').text.lower()
        if class_name not in class_map:
            continue
            
        class_id = class_map[class_name]
        bbox = obj.find('bndbox')
        
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)
        
        x_center = ((xmin + xmax) / 2) / img_width
        y_center = ((ymin + ymax) / 2) / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height
        
        yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    return yolo_annotations


class ProgressManager:
    def __init__(self, noise_levels):
        self.noise_levels = noise_levels
        self.progress_data = {level: {'annotations': 0, 'train': 0, 'val': 0} for level in noise_levels}
        self.max_values = {level: {'annotations': 0, 'train': 0, 'val': 0} for level in noise_levels}
        self.queue = Queue()
        self.should_stop = False

    def update_progress(self, noise_level, stage, current, total):
        self.queue.put(('update', noise_level, stage, current, total))

    def set_status(self, noise_level, status):
        self.queue.put(('status', noise_level, status))

    def stop(self):
        self.should_stop = True

def process_noise_level(noise_level, images_dir, annotations_dir, output_dir, train_indices, val_indices, force_regenerate, images, progress_queue):
    """Process images for a specific noise level with progress tracking."""
    try:
        dataset_dir = f"{output_dir}/dataset_{noise_level}"
        train_img_dir = f"{dataset_dir}/images/train"
        val_img_dir = f"{dataset_dir}/images/val"
        train_label_dir = f"{dataset_dir}/labels/train"
        val_label_dir = f"{dataset_dir}/labels/val"
        yaml_path = f"{dataset_dir}/dataset.yaml"

        # Check if dataset already exists
        if not force_regenerate and os.path.exists(dataset_dir):
            dirs_to_check = [train_img_dir, val_img_dir, train_label_dir, val_label_dir]
            if all(os.path.exists(d) and len(os.listdir(d)) > 0 for d in dirs_to_check) and os.path.exists(yaml_path):
                progress_queue.put(('status', noise_level, 'Skipped (already exists)'))
                return {'status': 'skipped', 'noise_level': noise_level}

        # Create directories
        os.makedirs(train_img_dir, exist_ok=True)
        os.makedirs(val_img_dir, exist_ok=True)
        os.makedirs(train_label_dir, exist_ok=True)
        os.makedirs(val_label_dir, exist_ok=True)

        # Process annotations
        xml_files = glob.glob(f"{annotations_dir}/*.xml")
        progress_queue.put(('max', noise_level, 'annotations', len(xml_files)))
        
        annotations_cache = {}
        for i, xml_path in enumerate(xml_files, 1):
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
                img_width = int(root.find('size/width').text)
                img_height = int(root.find('size/height').text)
                
                annotations_cache[Path(xml_path).stem] = {
                    'annotations': convert_xml_to_yolo(xml_path, img_width, img_height),
                    'width': img_width,
                    'height': img_height
                }
                progress_queue.put(('update', noise_level, 'annotations', i))
            except Exception as e:
                logging.warning(f"Failed to process annotation {xml_path}: {str(e)}")

        # Process training images
        progress_queue.put(('max', noise_level, 'train', len(train_indices)))
        for i, idx in enumerate(train_indices, 1):
            img_path = images[idx]
            img_stem = Path(img_path).stem
            
            output_img_path = f"{train_img_dir}/{Path(img_path).name}"
            add_noise(img_path, noise_level, output_img_path, seed=42+noise_level, preserve_channels=True)
            
            if img_stem in annotations_cache:
                output_label_path = f"{train_label_dir}/{img_stem}.txt"
                with open(output_label_path, 'w') as f:
                    f.write('\n'.join(annotations_cache[img_stem]['annotations']))
                    
            progress_queue.put(('update', noise_level, 'train', i))

        # Process validation images
        progress_queue.put(('max', noise_level, 'val', len(val_indices)))
        for i, idx in enumerate(val_indices, 1):
            img_path = images[idx]
            img_stem = Path(img_path).stem
            
            output_img_path = f"{val_img_dir}/{Path(img_path).name}"
            add_noise(img_path, noise_level, output_img_path, seed=42+noise_level, preserve_channels=True)
            
            if img_stem in annotations_cache:
                output_label_path = f"{val_label_dir}/{img_stem}.txt"
                with open(output_label_path, 'w') as f:
                    f.write('\n'.join(annotations_cache[img_stem]['annotations']))
                    
            progress_queue.put(('update', noise_level, 'val', i))

        # Create dataset.yaml
        yaml_content = {
            'path': os.path.abspath(dataset_dir),
            'train': 'images/train',
            'val': 'images/val',
            'names': {0: 'speedlimit', 1: 'stop', 2: 'crosswalk', 3: 'trafficlight'}
        }
        
        with open(f"{dataset_dir}/dataset.yaml", 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)

        progress_queue.put(('status', noise_level, 'Completed'))
        return {'status': 'success', 'noise_level': noise_level}

    except Exception as e:
        progress_queue.put(('status', noise_level, f'Error: {str(e)}'))
        return {'status': 'error', 'noise_level': noise_level, 'error': str(e)}

def generate_progress_table(progress_data, max_values):
    """Generate a rich table showing progress for all noise levels."""
    table = Table(show_header=True, header_style="bold magenta", expand=True)
    table.add_column("Noise Level")
    table.add_column("Annotations")
    table.add_column("Training")
    table.add_column("Validation")
    table.add_column("Status")

    for noise_level in sorted(progress_data.keys()):
        data = progress_data[noise_level]
        maxes = max_values[noise_level]
        
        def format_progress(current, total):
            if total == 0:
                return "-"
            percentage = (current / total) * 100 if total > 0 else 0
            return f"{percentage:.1f}% ({current}/{total})"

        table.add_row(
            f"{noise_level}%",
            format_progress(data['annotations'], maxes['annotations']),
            format_progress(data['train'], maxes['train']),
            format_progress(data['val'], maxes['val']),
            data.get('status', 'Starting...')
        )

    return table


class NoiseExperiment:
    def __init__(self, args):
        """Initialize the noise experiment."""
        self.images_dir = args.images_dir
        self.annotations_dir = args.annotations_dir
        self.experiment_dir = args.output_dir
        self.results_file = "experiment_results.csv"
        self.noise_levels = list(range(args.min_noise, args.max_noise + args.noise_step, args.noise_step))
        self.model_name = args.model
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.image_size = args.image_size
        self.preserve_channels = args.preserve_channels
        self.force_regenerate = args.force_regenerate
        
        # Compute optimal number of workers
        self.num_workers = min(8, max(1, mp.cpu_count() - 1))
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('noise_experiment.log'),
                logging.StreamHandler()
            ]
        )
        
        # Check CUDA availability and configure device
        if torch.cuda.is_available():
            self.device = 'cuda:0'
            gpu_name = torch.cuda.get_device_name(0)
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logging.info(f"Using GPU: {gpu_name}")
            logging.info(f"GPU Memory: {total_memory:.2f} GB")
            logging.info(f"CUDA Version: {torch.version.cuda}")
            
            # Log current GPU memory usage
            memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
            memory_cached = torch.cuda.memory_reserved(0) / 1024**3
            logging.info(f"Current GPU Memory Usage - Allocated: {memory_allocated:.2f} GB, Cached: {memory_cached:.2f} GB")
        else:
            self.device = 'cpu'
            cpu_count = mp.cpu_count()
            logging.info(f"CUDA not available. Using CPU with {cpu_count} cores")
            logging.info(f"CPU Device: {platform.processor()}")
            # Set number of threads for CPU training
            torch.set_num_threads(self.num_workers)
            logging.info(f"Using {self.num_workers} CPU threads for training")

    def generate_noisy_datasets(self):
        """Generate noisy versions of the dataset with parallel processing and live progress tracking."""
        logging.info("Starting dataset generation...")

        # Get all image files and create train/val split
        images = sorted(glob.glob(f"{self.images_dir}/*.png"))
        if not images:
            raise FileNotFoundError(f"No PNG images found in {self.images_dir}")

        indices = np.arange(len(images))
        np.random.seed(42)
        np.random.shuffle(indices)
        split_idx = int(len(indices) * 0.8)
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]

        # Initialize progress tracking
        progress_manager = ProgressManager(self.noise_levels)
        
        # Create process pool
        num_cores = max(1, mp.cpu_count() - 1) if max(1, mp.cpu_count() - 1) <= 60 else 60
        logging.info(f"Using {num_cores} CPU cores for parallel processing")

        processes = []
        for noise_level in self.noise_levels:
            p = mp.Process(
                target=process_noise_level,
                args=(noise_level, self.images_dir, self.annotations_dir, self.experiment_dir,
                    train_indices, val_indices, self.force_regenerate, images, progress_manager.queue)
            )
            processes.append(p)

        # Start progress display
        with Live(generate_progress_table(progress_manager.progress_data, progress_manager.max_values),
                refresh_per_second=4) as live:
            # Start all processes
            for p in processes:
                p.start()

            # Monitor progress queue and update display
            while any(p.is_alive() for p in processes):
                try:
                    while True:
                        msg = progress_manager.queue.get_nowait()
                        if msg[0] == 'update':
                            _, noise_level, stage, current = msg
                            progress_manager.progress_data[noise_level][stage] = current
                        elif msg[0] == 'max':
                            _, noise_level, stage, total = msg
                            progress_manager.max_values[noise_level][stage] = total
                        elif msg[0] == 'status':
                            _, noise_level, status = msg
                            progress_manager.progress_data[noise_level]['status'] = status
                        
                        # Update the display
                        live.update(generate_progress_table(
                            progress_manager.progress_data,
                            progress_manager.max_values
                        ))
                except Empty:
                    time.sleep(0.1)

            # Wait for all processes to complete
            for p in processes:
                p.join()

        # Final status update
        logging.info("Dataset generation completed.")

    def train_models(self, noise_levels=None, retrain_attempt=False):
        """
        Train YOLOv8 models for specified noise levels.
        
        Args:
            noise_levels (list, optional): Specific noise levels to train. If None, uses all levels.
            retrain_attempt (bool): Whether this is a retraining attempt for failed models.
        
        Returns:
            dict: Training results for each noise level
        """
        if noise_levels is None:
            noise_levels = self.noise_levels
            
        if not isinstance(noise_levels, list):
            noise_levels = [noise_levels]
        
        logging.info(f"{'Retraining' if retrain_attempt else 'Training'} models for noise levels: {noise_levels}")
        
        training_results = {}
        
        for noise_level in tqdm(noise_levels, desc="Training models"):
            retrain_start = datetime.now() if retrain_attempt else None
            
            try:
                model = YOLO(self.model_name)
                
                train_args = {
                    'data': f"{self.experiment_dir}/dataset_{noise_level}/dataset.yaml",
                    'epochs': self.epochs,
                    'imgsz': self.image_size,
                    'batch': self.batch_size,
                    'device': self.device,
                    'workers': self.num_workers,
                    'name': f"noise_{noise_level}",
                    'project': f"{self.experiment_dir}/models",
                    'patience': 50,
                    'optimizer': "auto",
                    'verbose': True,
                    'exist_ok': True,
                    'pretrained': True,
                    'close_mosaic': 10,
                    'seed': 42,
                    'val': True,
                    'rect': False,
                    'cos_lr': True
                }
                
                # CPU-specific optimizations
                if self.device == 'cpu':
                    train_args.update({
                        'batch': min(8, self.batch_size),
                        'amp': False,
                        'workers': max(1, self.num_workers // 2),
                        'plots': False
                    })
                else:
                    train_args.update({
                        'amp': True,
                    })
                
                # Remove invalid or deprecated arguments
                train_args = {k: v for k, v in train_args.items() if v is not None}
                
                results = model.train(**train_args)
                
                metrics = {
                    'noise_level': noise_level,
                    'mAP50': float(results.box.map50),
                    'mAP50-95': float(results.box.map),
                    'precision': float(results.box.mp),
                    'recall': float(results.box.mr)
                }
                
                if retrain_attempt:
                    retrain_end = datetime.now()
                    metrics.update({
                        'retrain_start': retrain_start.isoformat(),
                        'retrain_end': retrain_end.isoformat(),
                        'retrain_duration': str(retrain_end - retrain_start)
                    })
                    logging.info(f"Successfully retrained model for noise level {noise_level}%")
                    logging.info(f"Retraining details for noise level {noise_level}%:")
                    for key, value in metrics.items():
                        logging.info(f"  - {key}: {value}")
                
                training_results[noise_level] = metrics
                
            except Exception as e:
                error_msg = f"{'Retraining' if retrain_attempt else 'Training'} failed for noise level {noise_level}%: {str(e)}"
                logging.error(error_msg)
                training_results[noise_level] = {'error': str(e)}
                continue
                
            if self.device != 'cpu':
                torch.cuda.empty_cache()
        
        # Save training results
        results_file = f"{self.experiment_dir}/{'retrained' if retrain_attempt else 'training'}_results.csv"
        df = pd.DataFrame(list(training_results.values()))
        df.to_csv(results_file, index=False)
        
        return training_results

    def evaluate_models(self):
        """Evaluate models against all noise levels with automatic retraining for failures."""
        logging.info("Starting model evaluation...")
        
        results = []
        skipped_count = 0
        processed_count = 0

        for model_noise in tqdm(self.noise_levels, desc="Evaluating models"):
            model_path = f"{self.experiment_dir}/models/noise_{model_noise}/weights/best.pt"
            
            # Try to load the model
            try:
                model = YOLO(model_path)
            except Exception as e:
                # Log the failure and initiate retraining
                logging.error(f"Failed to load model for noise level {model_noise}%: {str(e)}")
                logging.warning(f"Attempting to retrain model for noise level {model_noise}%")
                
                # Attempt to retrain the model
                retrain_results = self.train_models(noise_levels=[model_noise], retrain_attempt=True)
                
                if retrain_results[model_noise].get('error'):
                    logging.error(f"Retraining also failed for noise level {model_noise}%, skipping evaluation")
                    continue
                    
                # Try to load the retrained model
                try:
                    model = YOLO(model_path)
                except Exception as e:
                    logging.error(f"Failed to load retrained model for noise level {model_noise}%, skipping evaluation")
                    continue
            
            # Process all validation noise levels for this model
            validation_results = []
            for val_noise in self.noise_levels:
                run_dir = f"{self.experiment_dir}/runs/detect/model_{model_noise}_val_{val_noise}"
                
                metrics = None
                if not self.force_regenerate and os.path.exists(run_dir):
                    metrics = get_existing_metrics(run_dir)
                    
                if metrics is not None:
                    skipped_count += 1
                    validation_results.append({
                        'model_noise': model_noise,
                        'validation_noise': val_noise,
                        **metrics
                    })
                else:
                    try:
                        val_results = model.val(
                            data=f"{self.experiment_dir}/dataset_{val_noise}/dataset.yaml",
                            device=self.device,
                            name=f"model_{model_noise}_val_{val_noise}",
                            save_json=True,
                            project=f"{self.experiment_dir}/runs/detect",
                            exist_ok=True
                        )
                        
                        processed_count += 1
                        validation_results.append({
                            'model_noise': model_noise,
                            'validation_noise': val_noise,
                            'mAP50-95': float(val_results.box.map),
                            'mAP50': float(val_results.box.map50),
                            'precision': float(val_results.box.mp),
                            'recall': float(val_results.box.mr)
                        })
                        
                    except Exception as e:
                        logging.error(f"Validation failed for model_{model_noise}_val_{val_noise}: {str(e)}")
                        continue
            
            # Extend results with successful validations
            results.extend(validation_results)
            
            # Clear GPU memory after processing each model
            if self.device != 'cpu':
                torch.cuda.empty_cache()

        # Save evaluation results
        if results:
            output_file = f"{self.experiment_dir}/evaluation_results.csv"
            new_df = pd.DataFrame(results)
            
            if os.path.exists(output_file) and not self.force_regenerate:
                # Merge with existing results
                existing_df = pd.read_csv(output_file)
                
                # Create unique identifiers for each run
                def create_run_id(row):
                    return f"{row['model_noise']}_{row['validation_noise']}"
                
                for df in [existing_df, new_df]:
                    df['run_id'] = df.apply(create_run_id, axis=1)
                
                # Update existing results with new ones
                combined_df = pd.concat([
                    existing_df[~existing_df['run_id'].isin(new_df['run_id'])],
                    new_df
                ]).drop('run_id', axis=1)
            else:
                combined_df = new_df
                
            # Sort and save results
            combined_df = combined_df.sort_values(['model_noise', 'validation_noise'])
            combined_df.to_csv(output_file, index=False)
            
            logging.info(f"Evaluation completed:")
            logging.info(f"  - Processed: {processed_count}")
            logging.info(f"  - Skipped: {skipped_count}")
            logging.info(f"  - Total results: {len(results)}")
            logging.info(f"Results saved to {output_file}")

    def run_experiment(self):
        """Run the complete experiment pipeline."""
        logging.info("Starting noise experiment...")
        start_time = datetime.now()
        
        # Create experiment directory
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        self.generate_noisy_datasets()
        self.train_models()
        self.evaluate_models()
        
        end_time = datetime.now()
        duration = end_time - start_time
        logging.info(f"Experiment completed in {duration}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run noise experiment with YOLOv8')
    
    # Dataset and output configuration
    parser.add_argument('--images-dir', type=str, default='images',
                    help='Path to the images directory')
    parser.add_argument('--annotations-dir', type=str, default='annotations',
                    help='Path to the annotations directory')
    parser.add_argument('--output-dir', type=str, default='noise_experiment',
                    help='Directory to store experiment results')
    
    # Model configuration
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                    choices=['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'],
                    help='YOLOv8 model to use')
    parser.add_argument('--image-size', type=int, default=640,
                    help='Image size for training')
    parser.add_argument('--batch-size', type=int, default=16,
                    help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train')
    
    # Noise configuration
    parser.add_argument('--min-noise', type=int, default=0,
                    help='Minimum noise percentage')
    parser.add_argument('--max-noise', type=int, default=95,
                    help='Maximum noise percentage')
    parser.add_argument('--noise-step', type=int, default=5,
                    help='Step size between noise levels')
    parser.add_argument('--preserve-channels', type=bool, default=True,
                    help='Preserve color channels when adding noise')
    
    # Training configuration
    parser.add_argument('--seed', type=int, default=42,
                    help='Random seed for reproducibility')
    parser.add_argument('--force-regenerate', action='store_true',
                    help='Force regeneration of noisy datasets even if they exist')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.min_noise < 0 or args.max_noise > 100:
        parser.error("Noise levels must be between 0 and 100")
    if args.min_noise >= args.max_noise:
        parser.error("min_noise must be less than max_noise")
    if args.noise_step <= 0:
        parser.error("noise_step must be positive")
        
    return args

if __name__ == "__main__":
    # Initialize Ultralytics settings
    from ultralytics.utils.checks import check_yolo as ultralytics_check
    ultralytics_check()
    
    # Parse command line arguments
    args = parse_args()
    
    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Initialize and run experiment
    experiment = NoiseExperiment(args)
    experiment.run_experiment()
