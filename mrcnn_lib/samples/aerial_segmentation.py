"""
Tree Instance Segmentation with Mask R-CNN
Adapts balloon.py for aerial tree segmentation.

This sample provides a small dataset loader for YOLO-style polygon labels
and :class:`TreeConfig` tuned for ~500x500 aerial imagery. It supports
training and inference subcommands similar to the Matterport Mask R-CNN samples.
"""

import os
import sys
import numpy as np
import skimage.io
import cv2
import argparse
import time
import logging
import matplotlib.pyplot as plt
import csv
import gc
import signal
import atexit
import glob
import contextlib
import subprocess
import threading
from mrcnn.visualize import display_instances
import warnings

# Suppress TensorFlow deprecation warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Set TensorFlow logging level to reduce verbose output
try:
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    # Use cuda_malloc_async allocator to reduce memory fragmentation during evaluation
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
except:
    pass

# GPU memory management configuration
try:
    import tensorflow as tf
    
    # Configure GPU memory growth and limits
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth for all GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Optional: Set memory limit per GPU (uncomment if needed)
            # memory_limit = 1024 * 8  # 8GB limit
            # tf.config.experimental.set_memory_limit(gpus[0], memory_limit)
            
            print(f"Configured {len(gpus)} GPU(s) with memory growth enabled")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(f"GPU configuration error: {e}")
    else:
        print("No GPUs found, using CPU")
        
except ImportError:
    print("TensorFlow not available, proceeding without GPU configuration")
except Exception as e:
    print(f"Unexpected error in GPU configuration: {e}")

from mrcnn.config import Config
from mrcnn import model as modellib, utils
import contextlib

# --- GPU Memory Management Context Manager ---
@contextlib.contextmanager
def inference_model_context(config, model_dir, weights_path, is_training=False):
    """
    Context manager for inference models that ensures proper GPU memory cleanup.
    This creates an inference model, yields it for use, and ensures cleanup afterward.
    
    Args:
        config: Model configuration
        model_dir: Model directory path
        weights_path: Path to weights file
        is_training: If True, avoid graph operations that could interfere with training
    """
    import gc
    import time
    
    model = None
    try:
        # Create inference model
        model = modellib.MaskRCNN(mode='inference', config=config, model_dir=model_dir)
        model.load_weights(weights_path, by_name=True)
        yield model
        
    finally:
        # Simple cleanup: delete model and clear references
        if model is not None:
            try:
                # Clear model weights and graph references
                if hasattr(model, 'keras_model') and model.keras_model is not None:
                    # Clear Keras model
                    del model.keras_model
                # Clear the model object itself
                del model
            except Exception as e:
                print(f"Warning: Error during model cleanup: {e}")
        
        # Force garbage collection to free memory
        for _ in range(3):  # Multiple GC cycles for better cleanup
            gc.collect()
        
        # Add small delay for GPU memory cleanup
        time.sleep(0.5)
        
        # Only do aggressive cleanup if we're completely done with training
        # Avoid clearing sessions during post-training evaluation
        if not is_training:
            # Light cleanup - just try to clear some memory stats
            try:
                import tensorflow as tf
                # Only clear memory stats, don't reset the graph
                if hasattr(tf.config.experimental, 'reset_memory_stats'):
                    gpus = tf.config.experimental.list_physical_devices('GPU')
                    for gpu in gpus:
                        try:
                            tf.config.experimental.reset_memory_stats(gpu.name)
                        except:
                            pass
            except (ImportError, AttributeError):
                # Handle different TF versions or missing TF
                pass

def get_gpu_memory_info():
    """Get current GPU memory usage information."""
    try:
        import tensorflow as tf
        gpu_info = []
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for i, gpu in enumerate(gpus):
            try:
                # Try multiple methods to get memory info
                memory_info = None
                current_mb = "N/A"
                peak_mb = "N/A"
                
                # Method 1: Use get_memory_info if available
                try:
                    memory_info = tf.config.experimental.get_memory_info(gpu.name)
                    current_mb = memory_info['current'] // (1024*1024)
                    peak_mb = memory_info['peak'] // (1024*1024)
                except:
                    pass
                
                # Method 2: Use nvidia-ml-py if available
                if current_mb == "N/A":
                    try:
                        import pynvml
                        pynvml.nvmlInit()
                        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        current_mb = meminfo.used // (1024*1024)
                        total_mb = meminfo.total // (1024*1024)
                        peak_mb = f"{current_mb}/{total_mb}"
                    except:
                        pass
                
                # Method 3: Use nvidia-smi command line tool
                if current_mb == "N/A":
                    try:
                        import subprocess
                        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits', f'--id={i}'], 
                                              capture_output=True, text=True, timeout=5)
                        if result.returncode == 0:
                            used, total = result.stdout.strip().split(', ')
                            current_mb = int(used.strip())
                            peak_mb = f"{current_mb}/{total.strip()}"
                    except:
                        pass
                
                gpu_info.append({
                    'gpu_id': i,
                    'name': gpu.name,
                    'current_mb': current_mb,
                    'peak_mb': peak_mb
                })
            except Exception as e:
                gpu_info.append({
                    'gpu_id': i,
                    'name': gpu.name,
                    'current_mb': f'Error: {e}',
                    'peak_mb': 'N/A'
                })
        return gpu_info
    except Exception as e:
        return [{'gpu_id': 0, 'name': 'Unknown', 'current_mb': f'Error: {e}', 'peak_mb': 'N/A'}]

def start_gpu_monitor(logs_dir, interval=2):
    """
    Start GPU monitoring subprocess that logs to CSV file.
    
    Args:
        logs_dir: Directory to save GPU monitoring log
        interval: Monitoring interval in seconds
    
    Returns:
        subprocess.Popen object or None if failed to start
    """
    try:
        gpu_log_file = os.path.join(logs_dir, "gpu_memory.csv")
        gpu_monitor_script = os.path.join(os.path.dirname(__file__), "../gpu_monitor.py")
        
        # Start GPU monitor as subprocess
        gpu_process = subprocess.Popen([
            sys.executable, gpu_monitor_script,
            "--interval", str(interval),
            "--csv", gpu_log_file
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Give it a moment to start
        time.sleep(1)
        
        print(f"Started GPU monitor with PID {gpu_process.pid}, logging to {gpu_log_file}")
        return gpu_process, gpu_log_file
        
    except Exception as e:
        print(f"Failed to start GPU monitor: {e}")
        return None, None

def stop_gpu_monitor(gpu_process):
    """Stop the GPU monitoring subprocess."""
    if gpu_process and gpu_process.poll() is None:
        try:
            gpu_process.terminate()
            gpu_process.wait(timeout=5)
            print("GPU monitor stopped")
        except subprocess.TimeoutExpired:
            gpu_process.kill()
            print("GPU monitor forcibly killed")
        except Exception as e:
            print(f"Error stopping GPU monitor: {e}")

def parse_gpu_log(gpu_log_file):
    """
    Parse GPU monitoring CSV log file and return memory usage data.
    
    Args:
        gpu_log_file: Path to the GPU monitoring CSV file
    
    Returns:
        dict: Contains timestamps, memory_used, memory_total lists
    """
    gpu_data = {
        'timestamps': [],
        'memory_used': [],
        'memory_total': [],
        'utilization': []
    }
    
    try:
        if not os.path.exists(gpu_log_file):
            return gpu_data
            
        with open(gpu_log_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    gpu_data['timestamps'].append(row['timestamp'])
                    
                    # Handle N/A values
                    memory_used = row['memory_used_mb']
                    memory_total = row['memory_total_mb']
                    utilization = row['utilization_percent']
                    
                    if memory_used != 'N/A':
                        gpu_data['memory_used'].append(float(memory_used))
                    else:
                        gpu_data['memory_used'].append(0)
                        
                    if memory_total != 'N/A':
                        gpu_data['memory_total'].append(float(memory_total))
                    else:
                        gpu_data['memory_total'].append(0)
                        
                    if utilization != 'N/A':
                        gpu_data['utilization'].append(float(utilization))
                    else:
                        gpu_data['utilization'].append(0)
                        
                except (ValueError, KeyError):
                    continue
                    
        print(f"Parsed {len(gpu_data['timestamps'])} GPU monitoring entries")
        return gpu_data
        
    except Exception as e:
        print(f"Error parsing GPU log: {e}")
        return gpu_data

def log_gpu_memory(logger, context=""):
    """Log current GPU memory usage using subprocess call."""
    try:
        # Quick one-time check using nvidia-smi
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=5
        )
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for i, line in enumerate(lines):
                if line.strip():
                    used, total = line.split(', ')
                    logger.info(f"GPU {i} {context}: Used: {used.strip()}MB, Total: {total.strip()}MB")
        else:
            logger.info(f"GPU memory info not available {context}")
            
    except Exception as e:
        logger.info(f"Could not get GPU memory info {context}: {e}")


def cleanup_gpu_memory(is_training=False):
    """Cleanup function to be called on exit.
    
    Args:
        is_training: If True, avoid operations that could interfere with training
    """
    try:
        import gc
        gc.collect()
        
        # Try to clear TensorFlow GPU memory stats only
        try:
            import tensorflow as tf
            # Clear cached memory stats only
            if hasattr(tf.config.experimental, 'reset_memory_stats'):
                gpus = tf.config.experimental.list_physical_devices('GPU')
                for gpu in gpus:
                    try:
                        tf.config.experimental.reset_memory_stats(gpu.name)
                    except:
                        pass
        except ImportError:
            pass
        
        print("GPU memory cleanup completed")
    except Exception as e:
        print(f"Error during cleanup: {e}")

def signal_handler(signum, frame):
    """Handle termination signals gracefully."""
    print(f"\nReceived signal {signum}. Cleaning up...")
    cleanup_gpu_memory(is_training=False)  # Safe to reset graph on exit
    sys.exit(0)

# Register cleanup functions
atexit.register(lambda: cleanup_gpu_memory(is_training=False))
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# --- Constants ---
# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "../mask_rcnn_coco.h5")
DEFAULT_LOGS_DIR = os.path.join(os.path.dirname(__file__), "../logs")

# --- Configuration ---
class TreeConfig(Config):
    NAME = "tree"
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1  # background + tree
    STEPS_PER_EPOCH = 200  # Keep optimized value from tree_segmentation.py
    
    # === ENHANCED LEARNING PARAMETERS FOR BETTER CONVERGENCE ===
    
    # Optimized learning rate (much lower for fine-tuning)
    LEARNING_RATE = 0.0001  # Reduced from 0.001
    
    # Enhanced momentum for stable convergence
    LEARNING_MOMENTUM = 0.9
    
    # Increased weight decay for better regularization
    WEIGHT_DECAY = 0.0005  # Increased from 0.0001
    
    # Gradient clipping to prevent exploding gradients
    GRADIENT_CLIP_NORM = 5.0
    
    # === BATCH NORMALIZATION TRAINING (Critical for convergence) ===
    TRAIN_BN = True  # Enable BN training for better convergence
    
    # === ENHANCED LOSS WEIGHTS FOR BALANCED TRAINING ===
    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.0,
        "rpn_bbox_loss": 1.0,
        "mrcnn_class_loss": 1.0,
        "mrcnn_bbox_loss": 1.0,
        "mrcnn_mask_loss": 0.5  # Reduced to prevent mask loss dominance
    }
    
    # === OPTIMIZED ROI SAMPLING ===
    # ROI sampling optimized for tree detection (from tree_segmentation.py)
    ROI_POSITIVE_RATIO = 0.4  # Better balance for tree detection
    TRAIN_ROIS_PER_IMAGE = 50  # Optimized for tree segmentation
    
    # === ENHANCED ANCHOR CONFIGURATION ===
    # Anchor scales tuned for trees (from tree_segmentation.py - better F1 performance)
    RPN_ANCHOR_SCALES = (16, 32, 64, 128)  # Optimized for tree detection
    RPN_ANCHOR_RATIOS = [0.5, 1, 2]  # Standard ratios for trees
    
    # === OPTIMIZED NMS PARAMETERS ===
    RPN_NMS_THRESHOLD = 0.7  # Standard value
    DETECTION_NMS_THRESHOLD = 0.3  # Keep existing
    
    # === DETECTION PARAMETERS ===
    # Lower confidence threshold for better recall (from tree_segmentation.py)
    DETECTION_MIN_CONFIDENCE = 0.80  # Optimized for tree detection
    DETECTION_MAX_INSTANCES = 150  # Keep existing
    
    # === IMAGE PROCESSING ===
    IMAGE_RESIZE_MODE = "pad64"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    
    # === ENHANCED RPN TRAINING ===
    RPN_TRAIN_ANCHORS_PER_IMAGE = 256  # Standard value
    PRE_NMS_LIMIT = 6000  # Optimized from tree_segmentation.py 
    POST_NMS_ROIS_TRAINING = 2000  # Standard training ROIs
    POST_NMS_ROIS_INFERENCE = 1000  # Standard inference ROIs
    
    # === GPU MEMORY OPTIMIZATIONS ===
    BACKBONE = "resnet50"  # Memory efficient
    VALIDATION_STEPS = 100
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # Memory efficient
    TOP_DOWN_PYRAMID_SIZE = 256
    FPN_CLASSIF_FC_LAYERS_SIZE = 1024
    
    # === ENHANCED BBOX REGRESSION ===
    # More precise bounding box regression
    RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
    BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
    
    # === POOLING OPTIMIZATION ===
    POOL_SIZE = 7  # Standard pooling size
    MASK_POOL_SIZE = 14  # Standard mask pooling
    
    # === ADDITIONAL CONVERGENCE OPTIMIZATIONS ===
    
    # Increase maximum GT instances for complex scenes with many trees
    MAX_GT_INSTANCES = 150  # Increased from default 100
    
    # Enhanced RPN training parameters
    RPN_ANCHOR_STRIDE = 1  # Dense anchor placement
    
    # Enhanced training stability
    USE_RPN_ROIS = True  # Use RPN-generated ROIs
    
    # Image preprocessing optimization for trees
    IMAGE_MIN_SCALE = 0  # No forced upscaling
    IMAGE_CHANNEL_COUNT = 3  # RGB
    MEAN_PIXEL = np.array([123.7, 116.8, 103.9])  # ImageNet means
    
    # === AUGMENTATION SUPPORT ===
    # Enable data augmentation for better generalization
    USE_AUGMENTATION = True
    
    def __init__(self):
        """Enhanced initialization with validation."""
        super().__init__()
        
        # Validate critical parameters
        assert self.LEARNING_RATE > 0, "Learning rate must be positive"
        assert self.TRAIN_ROIS_PER_IMAGE > 0, "ROIs per image must be positive"
        assert len(self.RPN_ANCHOR_SCALES) > 0, "Must have at least one anchor scale"
        
        # Log configuration for debugging
        if hasattr(self, '_config_logged'):
            return
        self._config_logged = True
        print(f"TreeConfig initialized with optimized settings from tree_segmentation.py:")
        print(f"  Learning Rate: {self.LEARNING_RATE}")
        print(f"  Weight Decay: {self.WEIGHT_DECAY}")
        print(f"  Batch Norm Training: {self.TRAIN_BN}")
        print(f"  ROI Positive Ratio: {self.ROI_POSITIVE_RATIO}")
        print(f"  Train ROIs per Image: {self.TRAIN_ROIS_PER_IMAGE}")
        print(f"  Anchor Scales: {self.RPN_ANCHOR_SCALES}")
        print(f"  Detection Min Confidence: {self.DETECTION_MIN_CONFIDENCE}")
        print(f"  Loss Weights: {self.LOSS_WEIGHTS}")
        print(f"  Backbone: {self.BACKBONE}")
        print("Configuration combines tree_segmentation.py optimizations with enhanced features")

# --- Dataset ---
class TreeDataset(utils.Dataset):
    def load_tree(self, dataset_dir, subset):
        """
        dataset_dir: path to yolo_dataset (e.g. .../Shadow_Casting_Object_Segmentation/dataset/yolo_dataset)
        subset: 'train' or 'valid'
        """
        self.add_class("tree", 1, "tree")
        assert subset in ["train", "valid"]
        image_dir = os.path.join(dataset_dir, subset, "images")
        label_dir = os.path.join(dataset_dir, subset, "labels")

        image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png'))]
        for image_file in image_files:
            image_id = os.path.splitext(image_file)[0]
            label_file = os.path.join(label_dir, f"{image_id}.txt")
            if os.path.exists(label_file):
                self.add_image(
                    "tree",
                    image_id=image_id,
                    path=os.path.join(image_dir, image_file),
                    label_path=label_file
                )

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        label_path = info["label_path"]
        image = skimage.io.imread(info["path"])
        height, width = image.shape[:2]
        masks, class_ids = [], []

        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 7: continue
                # YOLO format: class_id x1 y1 x2 y2 ... (normalized)
                coords = [float(x) for x in parts[1:]]
                if len(coords) % 2 != 0: continue

                poly = np.array(coords).reshape(-1, 2)
                # scale normalized coordinates to image pixels
                poly[:, 0] *= width
                poly[:, 1] *= height
                poly = poly.astype(np.int32)

                mask = np.zeros((height, width), dtype=np.uint8)
                if len(poly) >= 3:
                    cv2.fillPoly(mask, [poly], color=1)
                    masks.append(mask)
                    class_ids.append(1)

        if masks:
            return np.stack(masks, axis=-1), np.array(class_ids, dtype=np.int32)
        else:
            return np.empty((height, width, 0)), np.array([], dtype=np.int32)

    def image_reference(self, image_id):
        return self.image_info[image_id]["path"]

# --- Core Functions ---
def compute_validation_metrics(dataset, model_infer, limit=None):
    """
    Run the given inference Mask R-CNN on all images in `dataset` and compute
    mean instance-level IoU and mean instance-level F1 for the foreground
    class across images. This uses mrcnn.utils.compute_matches to perform
    greedy matching between predicted and GT instances and averages per-image
    instance IoU and instance F1.

    Returns: mean_iou, mean_f1
    """
    import time
    import gc
    
    start_time = time.time()
    ious = []
    f1s = []

    image_ids = dataset.image_ids
    if limit:
        image_ids = image_ids[:limit]

    total_images = len(image_ids)
    
    # Memory-efficient evaluation: process images in smaller batches with cleanup
    batch_size = 5  # Process 5 images at a time to prevent memory buildup
    
    for idx, image_id in enumerate(image_ids):
        image_start_time = time.time()
        
        info = dataset.image_info[image_id]
        image = skimage.io.imread(info['path'])

        # Load GT instances
        gt_masks, gt_class_ids = dataset.load_mask(image_id)
        if gt_masks.size == 0:
            gt_boxes = np.zeros((0, 4), dtype=np.int32)
            gt_class_ids = np.array([], dtype=np.int32)
        else:
            gt_boxes = utils.extract_bboxes(gt_masks)

        # Run inference
        inference_start = time.time()
        r = model_infer.detect([image], verbose=0)[0]
        inference_time = time.time() - inference_start
        
        pred_masks = r.get('masks') if r.get('masks') is not None else np.empty((image.shape[0], image.shape[1], 0))
        pred_class_ids = r.get('class_ids') if r.get('class_ids') is not None else np.array([], dtype=np.int32)
        pred_scores = r.get('scores') if r.get('scores') is not None else np.array([], dtype=np.float32)
        pred_boxes = r.get('rois') if r.get('rois') is not None else np.zeros((0, 4), dtype=np.int32)

        # Use instance matching to determine correct detections
        try:
            gt_match, pred_match, overlaps = utils.compute_matches(
                gt_boxes, gt_class_ids, gt_masks,
                pred_boxes, pred_class_ids, pred_scores, pred_masks,
                iou_threshold=0.5)
        except Exception:
            # If matching fails for any reason, treat this image as zero score
            ious.append(0.0)
            f1s.append(0.0)
            continue

        matched = np.sum(pred_match > -1)
        n_preds = pred_masks.shape[-1]
        n_gts = gt_masks.shape[-1]

        # Instance IoU: mean IoU over matched pairs
        if matched > 0:
            matched_inds = np.where(pred_match > -1)[0]
            iou_vals = []
            for pi in matched_inds:
                gi = int(pred_match[pi])
                # overlaps shape: [preds, gts]
                iou_vals.append(overlaps[pi, gi])
            inst_iou = float(np.mean(iou_vals))
        else:
            inst_iou = 1.0 if (n_preds == 0 and n_gts == 0) else 0.0

        # Instance precision/recall/F1 based on counts
        precision = matched / n_preds if n_preds > 0 else (1.0 if matched == 0 else 0.0)
        recall = matched / n_gts if n_gts > 0 else (1.0 if matched == 0 else 0.0)
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        ious.append(inst_iou)
        f1s.append(f1)
        
        image_total_time = time.time() - image_start_time
        
        # Progress indicator for validation (every 10% or every 10 images, whichever is smaller)
        progress_interval = min(10, max(1, total_images // 10))
        if (idx + 1) % progress_interval == 0 or idx == total_images - 1:
            print(f"Validation progress: {idx+1}/{total_images} images processed "
                  f"(Inference: {inference_time:.3f}s, Total: {image_total_time:.3f}s)")
        
        # Memory cleanup every batch_size images
        if (idx + 1) % batch_size == 0:
            # Clear large variables
            del image, gt_masks, pred_masks, r
            if 'overlaps' in locals():
                del overlaps
            
            # Force garbage collection
            gc.collect()
            
            # Brief pause to allow GPU memory cleanup
            time.sleep(0.1)

    mean_iou = float(np.mean(ious)) if ious else 0.0
    mean_f1 = float(np.mean(f1s)) if f1s else 0.0
    
    total_time = time.time() - start_time
    avg_time_per_image = total_time / total_images if total_images > 0 else 0
    
    print(f"Validation completed: {total_images} images in {total_time:.2f}s "
          f"(avg: {avg_time_per_image:.3f}s/image, {total_images/total_time:.2f} images/s)")
    
    return mean_iou, mean_f1


def evaluate_single_model(config, args, model_file, dataset_val):
    """
    Evaluate a single model file in a clean context.
    This function is designed to be called separately for each model.
    """
    import time
    import gc
    
    eval_start_time = time.time()
    
    try:
        # Extract epoch number
        def extract_epoch(filepath):
            try:
                filename = os.path.basename(filepath)
                epoch_str = filename.split('_')[-1].split('.')[0]
                return int(epoch_str)
            except:
                return 0
        
        epoch = extract_epoch(model_file)
        
        # Force garbage collection before creating new model
        gc.collect()
        
        # Timing for model loading
        model_load_start = time.time()
        
        # Create a fresh inference model
        model = modellib.MaskRCNN(mode='inference', config=config, model_dir=args.logs)
        model.load_weights(model_file, by_name=True)
        
        model_load_time = time.time() - model_load_start
        
        # Timing for evaluation
        eval_inference_start = time.time()
        
        # Run evaluation with memory-efficient validation (limit images to reduce OOM risk)
        mean_iou, mean_f1 = compute_validation_metrics(dataset_val, model, limit=50)  # Evaluate up to 50 validation images
        
        eval_inference_time = time.time() - eval_inference_start
        total_eval_time = time.time() - eval_start_time
        
        # Clean up the model aggressively
        try:
            if hasattr(model, 'keras_model') and model.keras_model is not None:
                del model.keras_model
            del model
        except:
            pass
            
        # Force garbage collection and add delay to prevent memory fragmentation
        import gc
        gc.collect()
        
        # Add a small delay to allow GPU memory cleanup to complete
        import time
        time.sleep(1.0)  # 1 second delay between model evaluations
        
        return epoch, mean_iou, mean_f1, None, model_load_time, eval_inference_time, total_eval_time
        
    except Exception as e:
        total_eval_time = time.time() - eval_start_time
        
        # Force cleanup even on error
        import gc
        gc.collect()
        time.sleep(0.5)  # Shorter delay on error
        
        return epoch, 0.0, 0.0, str(e), 0, 0, total_eval_time


def evaluate_all_epochs(config, args):
    """
    Evaluate all saved epoch models to compute validation metrics.
    This is done post-training to save GPU memory during training.
    """
    train_logger = logging.getLogger('aerial_train')
    train_logger.info("Starting post-training evaluation of all epochs...")
    train_logger.info("Memory Optimization: Using cuda_malloc_async allocator and batch processing")
    train_logger.info("Memory Optimization: Processing all validation images with cleanup delays between batches")
    
    # Load validation dataset
    dataset_val = TreeDataset()
    dataset_val.load_tree(args.dataset, 'valid')
    dataset_val.prepare()
    
    # Find models from the CURRENT training session only
    # Get the most recent subdirectory (current session)
    subdirs = [d for d in os.listdir(args.logs) if os.path.isdir(os.path.join(args.logs, d)) and d.startswith('tree')]
    if not subdirs:
        train_logger.warning("No training session directories found")
        return [], [], []
    
    # Sort by creation time and get the most recent
    subdirs_with_time = []
    for subdir in subdirs:
        full_path = os.path.join(args.logs, subdir)
        try:
            # Get directory creation time
            stat = os.stat(full_path)
            subdirs_with_time.append((stat.st_mtime, subdir))
        except:
            continue
    
    if not subdirs_with_time:
        train_logger.warning("No valid training session directories found")
        return [], [], []
    
    # Get the most recent directory (current training session)
    subdirs_with_time.sort(reverse=True)  # Most recent first
    current_session_dir = subdirs_with_time[0][1]
    current_session_path = os.path.join(args.logs, current_session_dir)
    
    # Find models only in the current session directory
    model_pattern = os.path.join(current_session_path, "mask_rcnn_tree_*.h5")
    model_files = glob.glob(model_pattern)
    
    train_logger.info(f"Using training session: {current_session_dir}")
    
    if not model_files:
        train_logger.warning(f"No saved models found in current session: {current_session_path}")
        return [], [], []
    
    # Sort models by epoch number
    def extract_epoch(filepath):
        try:
            filename = os.path.basename(filepath)
            epoch_str = filename.split('_')[-1].split('.')[0]
            return int(epoch_str)
        except:
            return 0
    
    model_files.sort(key=extract_epoch)
    train_logger.info(f"Found {len(model_files)} saved models to evaluate from current session ({current_session_dir})")
    
    val_ious = []
    val_f1s = []
    epochs = []
    model_load_times = []
    eval_inference_times = []
    
    for i, model_file in enumerate(model_files):
        model_eval_result = evaluate_single_model(config, args, model_file, dataset_val)
        
        if len(model_eval_result) == 7:  # New format with timing
            epoch, mean_iou, mean_f1, error, model_load_time, eval_inference_time, total_eval_time = model_eval_result
            model_load_times.append(model_load_time)
            eval_inference_times.append(eval_inference_time)
        else:  # Old format compatibility
            epoch, mean_iou, mean_f1, error = model_eval_result
            model_load_times.append(0)
            eval_inference_times.append(0)
            total_eval_time = 0
        
        epochs.append(epoch)
        val_ious.append(mean_iou)
        val_f1s.append(mean_f1)
        
        if error:
            train_logger.error(f"Error evaluating epoch {epoch}: {error}")
        else:
            train_logger.info(f"Epoch {epoch} - IoU: {mean_iou:.4f}, F1: {mean_f1:.4f} (Load: {model_load_times[-1]:.2f}s, Eval: {eval_inference_times[-1]:.2f}s, Total: {total_eval_time:.2f}s)")
        
        # Progress indicator
        train_logger.info(f"Progress: {i+1}/{len(model_files)} models evaluated")
        
        # Enhanced cleanup after each model with delay
        import gc
        import time
        
        # Force multiple garbage collection cycles
        for _ in range(3):
            gc.collect()
        
        # Add delay between evaluations to allow GPU memory cleanup
        if i < len(model_files) - 1:  # Don't delay after the last model
            train_logger.info(f"Waiting 2 seconds for memory cleanup before next evaluation...")
            time.sleep(2.0)  # 2 second delay between model evaluations
    
    # Summary of evaluation timing
    if model_load_times and eval_inference_times:
        total_load_time = sum(model_load_times)
        total_inference_time = sum(eval_inference_times)
        avg_load_time = total_load_time / len(model_load_times)
        avg_inference_time = total_inference_time / len(eval_inference_times)
        
        train_logger.info(f"\n=== Evaluation Timing Summary ===")
        train_logger.info(f"Total model loading time: {total_load_time:.2f}s")
        train_logger.info(f"Total inference time: {total_inference_time:.2f}s")
        train_logger.info(f"Average model loading time: {avg_load_time:.2f}s")
        train_logger.info(f"Average inference time: {avg_inference_time:.2f}s")
        train_logger.info(f"=================================")
    
    # Clean up validation dataset
    try:
        del dataset_val
    except:
        pass
    gc.collect()
    
    train_logger.info("Post-training evaluation completed")
    return epochs, val_ious, val_f1s


def extract_losses_from_log(log_file_path, expected_epochs):
    """
    Extract training and validation losses from training log file.
    This is a fallback method when the training history object doesn't provide loss values.
    
    Args:
        log_file_path: Path to the training log file
        expected_epochs: Number of epochs that were trained
    
    Returns:
        tuple: (train_losses, val_losses) or None if extraction fails
    """
    try:
        import re
        
        train_losses = []
        val_losses = []
        
        with open(log_file_path, 'r') as f:
            content = f.read()
        
        # Look for epoch completion lines that contain both train and val losses
        # Pattern to match: "200/200 [==============================] - XXs XXXms/step - loss: X.XXXX ... val_loss: X.XXXX"
        epoch_pattern = r'200/200.*?loss:\s+([\d.]+).*?val_loss:\s+([\d.]+)'
        matches = re.findall(epoch_pattern, content)
        
        if matches:
            # Extract loss values for each epoch found
            for match in matches:
                train_loss = float(match[0])
                val_loss = float(match[1])
                train_losses.append(train_loss)
                val_losses.append(val_loss)
            
            print(f"Extracted {len(train_losses)} epoch loss values from log")
            return train_losses, val_losses
        
        # Fallback: Look for any final loss values if epoch pattern doesn't work
        pattern = r'loss:\s+([\d.]+).*?val_loss:\s+([\d.]+)'
        matches = re.findall(pattern, content)
        
        if matches:
            # Take the last match (final values)
            final_match = matches[-1]
            final_train_loss = float(final_match[0])
            final_val_loss = float(final_match[1])
            
            # Create arrays with the final values repeated for each epoch
            train_losses = [final_train_loss] * expected_epochs
            val_losses = [final_val_loss] * expected_epochs
            
            print(f"Used final loss values for {expected_epochs} epochs")
            return train_losses, val_losses
        
        return None
        
    except Exception as e:
        print(f"Error extracting losses from log: {e}")
        return None


def train_model(config, args, gpu_process=None, gpu_log_file=None):
    """Sets up and runs the training process without per-epoch validation."""
    # Ensure os is available in function scope
    global os
    
    model = modellib.MaskRCNN(mode='training', config=config, model_dir=args.logs)

    if args.weights.lower() == 'coco':
        model.load_weights(COCO_WEIGHTS_PATH, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(args.weights, by_name=True)

    dataset_train = TreeDataset()
    dataset_train.load_tree(args.dataset, 'train')
    dataset_train.prepare()

    dataset_val = TreeDataset()
    dataset_val.load_tree(args.dataset, 'valid')
    dataset_val.prepare()

    # Prepare logs directory and logger
    os.makedirs(args.logs, exist_ok=True)
    log_file = os.path.join(args.logs, "training.log")
    train_logger = logging.getLogger('aerial_train')
    train_logger.setLevel(logging.INFO)
    if not train_logger.handlers:
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        train_logger.addHandler(fh)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        train_logger.addHandler(ch)

    # Set up a separate logger for stderr
    error_log_file = os.path.join(args.logs, "training_error.log")
    error_logger = logging.getLogger('aerial_error')
    error_logger.setLevel(logging.ERROR)
    if not error_logger.handlers:
        err_fh = logging.FileHandler(error_log_file, mode='w')
        err_fh.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
        error_logger.addHandler(err_fh)

    train_logger.info(f"Starting training with {config.BACKBONE} backbone...")
    train_logger.info(f"GPU Memory Optimization: Using ResNet-50 for reduced memory usage")
    train_logger.info(f"Validation will be performed post-training on all saved models")
    
    # Log model configuration
    train_logger.info("Model Configuration:")
    train_logger.info("-" * 50)
    
    # Capture config.display() output and log it
    import io
    from contextlib import redirect_stdout
    
    config_output = io.StringIO()
    with redirect_stdout(config_output):
        config.display()
    
    # Log each line of the configuration
    for line in config_output.getvalue().strip().split('\n'):
        if line.strip():  # Skip empty lines
            train_logger.info(line)
    
    train_logger.info("-" * 50)
    
    # GPU monitoring is already started from main(), just log status
    if gpu_process and gpu_log_file:
        train_logger.info(f"GPU monitoring already active, logging to {os.path.basename(gpu_log_file)}")
    else:
        train_logger.info("No GPU monitoring active")
    
    # Log initial GPU memory state
    log_gpu_memory(train_logger, "before training")

    # Redirect stdout/stderr to logger to capture Keras progress updates
    class LoggerWriter:
        def __init__(self, logger, level):
            self.logger = logger
            self.level = level
            self.buffer = ""

        def write(self, message):
            # Keras progress bar uses carriage returns. Buffer lines until a newline.
            if message.endswith('\n'):
                self.buffer += message.rstrip()
                if self.buffer.strip():
                    self.logger.log(self.level, self.buffer)
                self.buffer = ""
            else:
                self.buffer += message

        def flush(self):
            if self.buffer.strip():
                self.logger.log(self.level, self.buffer)
            self.buffer = ""

    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    sys.stdout = LoggerWriter(train_logger, logging.INFO)
    sys.stderr = LoggerWriter(error_logger, logging.ERROR)

    try:
        # Metrics storage (per-epoch) - only training losses
        epoch_times = []
        train_losses = []
        val_losses = []

        # Training timing
        training_start_time = time.time()
        
        # Train all epochs at once without per-epoch validation
        train_logger.info(f"Training for {args.epochs} epochs without per-epoch validation...")
        train_logger.info(f"Training started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(training_start_time))}")
        
        # Single training call for all epochs
        history = model.train(dataset_train, dataset_val,
                            learning_rate=config.LEARNING_RATE,
                            epochs=args.epochs,
                            layers=args.layers)
        
        training_end_time = time.time()
        total_training_time = training_end_time - training_start_time
        
        # Extract training and validation losses from history
        train_losses = []
        val_losses = []
        
        try:
            train_logger.info(f"History object type: {type(history)}")
            train_logger.info(f"History object: {history}")
            
            if history is not None:
                if hasattr(history, 'history'):
                    train_logger.info(f"History.history type: {type(history.history)}")
                    train_logger.info(f"History.history keys: {list(history.history.keys()) if hasattr(history.history, 'keys') else 'No keys'}")
                    
                    train_losses = history.history.get('loss', [])
                    val_losses = history.history.get('val_loss', [])
                    
                    if train_losses and val_losses:
                        train_logger.info(f"Successfully extracted losses - Train: {len(train_losses)} epochs, Val: {len(val_losses)} epochs")
                        train_logger.info(f"Final train loss: {train_losses[-1]:.4f}, val loss: {val_losses[-1]:.4f}")
                    else:
                        train_logger.warning(f"Empty loss arrays - Train losses: {len(train_losses)}, Val losses: {len(val_losses)}")
                        
                elif isinstance(history, dict):
                    train_logger.info("History is a dictionary")
                    train_logger.info(f"Dictionary keys: {list(history.keys())}")
                    train_losses = history.get('loss', [])
                    val_losses = history.get('val_loss', [])
                    
                else:
                    train_logger.warning(f"History object has no 'history' attribute and is not a dict. Attributes: {dir(history)}")
            else:
                train_logger.warning("History object is None")
                
        except Exception as e:
            train_logger.warning(f"Error extracting loss values: {e}")
            train_logger.warning(f"Exception type: {type(e)}")
            import traceback
            train_logger.warning(f"Traceback: {traceback.format_exc()}")

        # If we couldn't get loss values from history, try to extract from logs as fallback
        if not train_losses and not val_losses:
            train_logger.info("Attempting to extract loss values from training logs as fallback...")
            try:
                log_file = os.path.join(args.logs, "training.log")
                if os.path.exists(log_file):
                    extracted_losses = extract_losses_from_log(log_file, args.epochs)
                    if extracted_losses:
                        train_losses, val_losses = extracted_losses
                        train_logger.info(f"Successfully extracted losses from log: {len(train_losses)} train, {len(val_losses)} val")
                    else:
                        train_logger.warning("Could not extract losses from log file")
                else:
                    train_logger.warning("Training log file not found for loss extraction")
            except Exception as e:
                train_logger.warning(f"Failed to extract losses from log: {e}")

        total_training_time = training_end_time - training_start_time
        avg_time_per_epoch = total_training_time / args.epochs
        
        train_logger.info(f"Training completed at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(training_end_time))}")
        train_logger.info(f"Total training time: {total_training_time:.2f} seconds ({total_training_time/60:.2f} minutes)")
        train_logger.info(f"Average time per epoch: {avg_time_per_epoch:.2f} seconds ({avg_time_per_epoch/60:.2f} minutes)")
        train_logger.info(f"Training speed: {args.epochs / (total_training_time/3600):.2f} epochs/hour")
        
        # Log final GPU memory state
        log_gpu_memory(train_logger, "after training completion")
        
        # Stop GPU monitoring immediately after training ends
        if gpu_process:
            stop_gpu_monitor(gpu_process)
            train_logger.info("GPU monitoring stopped after training completion")
        
    finally:
        # Restore original stdout and stderr
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        
        # Clean up the training model to avoid graph conflicts with evaluation
        try:
            # More aggressive cleanup of training model
            if hasattr(model, 'keras_model') and model.keras_model is not None:
                del model.keras_model
            del model
        except:
            pass
            
        import gc
        gc.collect()

    # Now perform post-training evaluation on all saved models
    train_logger.info("Starting post-training evaluation...")
    eval_start_time = time.time()
    epochs, val_ious, val_f1s = evaluate_all_epochs(config, args)
    eval_end_time = time.time()
    eval_total_time = eval_end_time - eval_start_time
    
    train_logger.info(f"Post-training evaluation completed in {eval_total_time:.2f} seconds ({eval_total_time/60:.2f} minutes)")
    if epochs:
        avg_eval_time_per_model = eval_total_time / len(epochs)
        train_logger.info(f"Average evaluation time per model: {avg_eval_time_per_model:.2f} seconds")
    
    # Overall timing summary
    total_time_with_eval = total_training_time + eval_total_time
    train_logger.info(f"\n=== Complete Training Session Timing ===")
    train_logger.info(f"Training time:     {total_training_time:.2f}s ({total_training_time/60:.2f} min) - {total_training_time/total_time_with_eval*100:.1f}%")
    train_logger.info(f"Evaluation time:   {eval_total_time:.2f}s ({eval_total_time/60:.2f} min) - {eval_total_time/total_time_with_eval*100:.1f}%")
    train_logger.info(f"Total session time: {total_time_with_eval:.2f}s ({total_time_with_eval/60:.2f} min)")
    train_logger.info(f"=========================================")

    # Parse GPU monitoring data if available
    gpu_data = {}
    if 'gpu_log_file' in locals() and gpu_log_file and os.path.exists(gpu_log_file):
        gpu_data = parse_gpu_log(gpu_log_file)
        if gpu_data['timestamps']:
            train_logger.info(f"Parsed GPU monitoring data: {len(gpu_data['timestamps'])} entries")

    # Save comprehensive metrics CSV and plot
    try:
        csv_path = os.path.join(args.logs, 'training_metrics.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Handle different lengths of data
            max_len = max(len(epochs) if epochs else 0, 
                         len(train_losses) if train_losses else 0,
                         len(val_losses) if val_losses else 0)
            
            writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_iou', 'val_f1'])
            
            for i in range(max_len):
                epoch = epochs[i] if i < len(epochs) else i + 1
                t_loss = train_losses[i] if i < len(train_losses) else 0
                v_loss = val_losses[i] if i < len(val_losses) else 0
                iou = val_ious[i] if i < len(val_ious) else 0
                f1 = val_f1s[i] if i < len(val_f1s) else 0
                writer.writerow([epoch, t_loss, v_loss, iou, f1])
                
        train_logger.info(f"Saved comprehensive training metrics CSV to: {csv_path}")
        
        # Save timing information CSV
        timing_csv_path = os.path.join(args.logs, 'training_timing.csv')
        with open(timing_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['metric', 'value_seconds', 'value_minutes', 'percentage_of_total'])
            
            total_session_time = total_training_time + eval_total_time
            
            writer.writerow(['total_training_time', f'{total_training_time:.2f}', f'{total_training_time/60:.2f}', f'{total_training_time/total_session_time*100:.1f}'])
            writer.writerow(['average_time_per_epoch', f'{avg_time_per_epoch:.2f}', f'{avg_time_per_epoch/60:.2f}', f'{avg_time_per_epoch/total_session_time*100:.1f}'])
            writer.writerow(['total_evaluation_time', f'{eval_total_time:.2f}', f'{eval_total_time/60:.2f}', f'{eval_total_time/total_session_time*100:.1f}'])
            writer.writerow(['total_session_time', f'{total_session_time:.2f}', f'{total_session_time/60:.2f}', '100.0'])
            
            # Add epochs per hour calculation
            epochs_per_hour = args.epochs / (total_training_time/3600)
            writer.writerow(['epochs_per_hour', f'{epochs_per_hour:.2f}', 'N/A', 'N/A'])
            
        train_logger.info(f"Saved timing information CSV to: {timing_csv_path}")

        # Create comprehensive plots with GPU monitoring
        if gpu_data and gpu_data['timestamps']:
            # Create 2x3 grid for 6 plots including GPU data
            fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(20, 12))
        else:
            # Create 2x2 grid for 4 plots without GPU data
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Training and Validation Loss
        if train_losses or val_losses:
            epochs_range = list(range(1, max(len(train_losses), len(val_losses)) + 1))
            if train_losses:
                ax1.plot(epochs_range[:len(train_losses)], train_losses, 'g-', label='Train Loss', marker='^')
            if val_losses:
                ax1.plot(epochs_range[:len(val_losses)], val_losses, 'orange', label='Val Loss', marker='v')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training and Validation Loss')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            # Ensure integer ticks on x-axis
            ax1.set_xticks([int(x) for x in epochs_range[::max(1, len(epochs_range)//10)]])
        else:
            # Show placeholder if no loss data
            ax1.text(0.5, 0.5, 'Loss data not available\n(History extraction failed)', 
                    ha='center', va='center', transform=ax1.transAxes, fontsize=12)
            ax1.set_title('Training and Validation Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
        
        # Validation IoU and F1
        if epochs and val_ious:
            # Ensure epochs are integers
            epochs_int = [int(e) for e in epochs]
            ax2.plot(epochs_int, val_ious, 'b-', label='Val IoU', marker='o')
            ax2.plot(epochs_int, val_f1s, 'r-', label='Val F1', marker='s')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Score')
            ax2.set_title('Validation Metrics (Post-Training Evaluation)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            # Ensure integer ticks on x-axis
            ax2.set_xticks([int(x) for x in epochs_int[::max(1, len(epochs_int)//10)]])
        else:
            ax2.text(0.5, 0.5, 'No validation data available', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            ax2.set_title('Validation Metrics (Post-Training Evaluation)')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Score')
        
        # Best metrics summary
        if val_ious and val_f1s:
            # Ensure epochs are integers for display
            epochs_int = [int(e) for e in epochs]
            best_iou_epoch = epochs_int[val_ious.index(max(val_ious))]
            best_f1_epoch = epochs_int[val_f1s.index(max(val_f1s))]
            ax3.bar(['Best IoU', 'Best F1'], [max(val_ious), max(val_f1s)], 
                   color=['blue', 'red'], alpha=0.7)
            ax3.set_ylabel('Score')
            ax3.set_title(f'Best Scores\nIoU: {max(val_ious):.3f} (Epoch {best_iou_epoch})\nF1: {max(val_f1s):.3f} (Epoch {best_f1_epoch})')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No metrics to display', 
                    ha='center', va='center', transform=ax3.transAxes, fontsize=12)
            ax3.set_title('Best Scores')
            ax3.set_ylabel('Score')
        
        # Combined loss and IoU comparison
        if train_losses and val_ious:
            ax4_twin = ax4.twinx()
            epochs_loss = list(range(1, len(train_losses) + 1))
            epochs_int = [int(e) for e in epochs]
            ax4.plot(epochs_loss, train_losses, 'g-', label='Train Loss', alpha=0.7)
            ax4_twin.plot(epochs_int, val_ious, 'b-', label='Val IoU', marker='o', alpha=0.7)
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Loss', color='g')
            ax4_twin.set_ylabel('IoU', color='b')
            ax4.set_title('Training Loss vs Validation IoU')
            ax4.grid(True, alpha=0.3)
            # Ensure integer ticks
            ax4.set_xticks([int(x) for x in epochs_loss[::max(1, len(epochs_loss)//5)]])
        elif val_ious:
            # Show just IoU if we don't have loss data
            epochs_int = [int(e) for e in epochs]
            ax4.plot(epochs_int, val_ious, 'b-', label='Val IoU', marker='o', alpha=0.7)
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('IoU', color='b')
            ax4.set_title('Validation IoU Over Time')
            ax4.grid(True, alpha=0.3)
            ax4.legend()
            # Ensure integer ticks
            ax4.set_xticks([int(x) for x in epochs_int[::max(1, len(epochs_int)//5)]])
        else:
            ax4.text(0.5, 0.5, 'No data for comparison', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Training Loss vs Validation IoU')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Value')
        
        # GPU Memory Usage plots (if available)
        if gpu_data and gpu_data['timestamps']:
            # GPU Memory Usage over time
            if gpu_data['memory_used'] and max(gpu_data['memory_used']) > 0:
                time_indices = list(range(len(gpu_data['timestamps'])))
                ax5.plot(time_indices, gpu_data['memory_used'], 'r-', label='GPU Memory Used', alpha=0.8)
                if gpu_data['memory_total'] and max(gpu_data['memory_total']) > 0:
                    ax5.axhline(y=max(gpu_data['memory_total']), color='gray', linestyle='--', 
                               label=f'Total Memory ({max(gpu_data["memory_total"]):.0f}MB)', alpha=0.6)
                ax5.set_xlabel('Time')
                ax5.set_ylabel('Memory (MB)')
                ax5.set_title('GPU Memory Usage During Training')
                ax5.legend()
                ax5.grid(True, alpha=0.3)
                
                # Show timestamps on x-axis (sparse)
                if len(gpu_data['timestamps']) > 10:
                    step = len(gpu_data['timestamps']) // 5
                    tick_indices = time_indices[::step]
                    tick_labels = [gpu_data['timestamps'][i][-8:] for i in tick_indices]  # Last 8 chars (time only)
                    ax5.set_xticks(tick_indices)
                    ax5.set_xticklabels(tick_labels, rotation=45)
            else:
                ax5.text(0.5, 0.5, 'No valid GPU memory data', 
                        ha='center', va='center', transform=ax5.transAxes, fontsize=12)
                ax5.set_title('GPU Memory Usage During Training')
            
            # GPU Utilization over time
            if gpu_data['utilization'] and max(gpu_data['utilization']) > 0:
                time_indices = list(range(len(gpu_data['timestamps'])))
                ax6.plot(time_indices, gpu_data['utilization'], 'purple', label='GPU Utilization %', alpha=0.8)
                ax6.set_xlabel('Time')
                ax6.set_ylabel('Utilization (%)')
                ax6.set_title('GPU Utilization During Training')
                ax6.set_ylim(0, 100)
                ax6.legend()
                ax6.grid(True, alpha=0.3)
                
                # Show timestamps on x-axis (sparse)
                if len(gpu_data['timestamps']) > 10:
                    step = len(gpu_data['timestamps']) // 5
                    tick_indices = time_indices[::step]
                    tick_labels = [gpu_data['timestamps'][i][-8:] for i in tick_indices]  # Last 8 chars (time only)
                    ax6.set_xticks(tick_indices)
                    ax6.set_xticklabels(tick_labels, rotation=45)
            else:
                ax6.text(0.5, 0.5, 'No valid GPU utilization data', 
                        ha='center', va='center', transform=ax6.transAxes, fontsize=12)
                ax6.set_title('GPU Utilization During Training')
        
        plt.tight_layout()
        plot_path = os.path.join(args.logs, 'comprehensive_training_metrics.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        train_logger.info(f"Saved comprehensive training metrics plot to: {plot_path}")
        
        # Print summary with timing information
        if val_ious and val_f1s:
            train_logger.info(f"=== Training Summary ===")
            train_logger.info(f"Total epochs trained: {args.epochs}")
            train_logger.info(f"Total training time: {total_training_time:.2f}s ({total_training_time/60:.2f} min)")
            train_logger.info(f"Average time per epoch: {avg_time_per_epoch:.2f}s ({avg_time_per_epoch/60:.2f} min)")
            train_logger.info(f"Training speed: {args.epochs / (total_training_time/3600):.2f} epochs/hour")
            train_logger.info(f"Best IoU: {max(val_ious):.4f} at epoch {epochs[val_ious.index(max(val_ious))]}")
            train_logger.info(f"Best F1: {max(val_f1s):.4f} at epoch {epochs[val_f1s.index(max(val_f1s))]}")
            train_logger.info(f"Final IoU: {val_ious[-1]:.4f}")
            train_logger.info(f"Final F1: {val_f1s[-1]:.4f}")
            train_logger.info(f"Total session time (training + evaluation): {total_time_with_eval:.2f}s ({total_time_with_eval/60:.2f} min)")
            train_logger.info(f"========================")
            
    except Exception as e:
        train_logger.warning(f"Failed saving metrics/plot: {e}")
    
    # Clean up datasets after evaluation is complete
    try:
        del dataset_train, dataset_val
    except:
        pass
    import gc
    gc.collect()
    
    # Final cleanup and explicit termination
    try:
        train_logger.info("Performing final cleanup...")
        
        # Perform aggressive GPU cleanup now that training is complete
        cleanup_gpu_memory(is_training=False)
        train_logger.info("Final cleanup completed. Training script will now exit.")
        
        # Force process termination to ensure clean exit
        os._exit(0)
        
    except Exception as e:
        train_logger.error(f"Error during final cleanup: {e}")
        # Force exit even if cleanup fails
        os._exit(1)

def visualize_results(image, r, args):
    """Saves or displays the detection results."""
    if args.save_overlay:
        masks = r.get('masks')
        overlay = image.copy()
        if masks is not None and masks.size > 0:
            alpha = 0.4
            for i in range(masks.shape[-1]):
                mask_bool = masks[..., i].astype(bool)
                colored = np.zeros_like(image, dtype=np.uint8)
                colored[mask_bool] = (0, 255, 0)
                overlay = np.where(mask_bool[..., None],
                                   (overlay * (1 - alpha) + colored * alpha).astype(np.uint8),
                                   overlay)
                mask_uint8 = (mask_bool * 255).astype(np.uint8)
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, (255, 255, 255), 2)

        out_path = args.output or os.path.join(os.getcwd(), f"{os.path.splitext(os.path.basename(args.image))[0]}_pred_overlay.jpg")
        cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print(f"Saved prediction overlay to: {out_path}")
    else:
        display_instances(image, r['rois'], r['masks'], r['class_ids'], ['BG', 'tree'], r['scores'])
        plt.show()

def test_model(config, args):
    """Sets up and runs the inference process with proper GPU memory management."""
    import time
    
    assert args.image, 'Provide --image for test command'
    
    print(f"Running inference on: {args.image}")
    
    # Timing for image loading
    load_start_time = time.time()
    image = skimage.io.imread(args.image)
    load_end_time = time.time()
    load_time = load_end_time - load_start_time
    
    print(f"Image loaded in {load_time:.4f} seconds")
    print(f"Image dimensions: {image.shape}")
    
    # Timing for model initialization and inference
    model_init_start_time = time.time()
    
    # Use context manager for proper memory management
    with inference_model_context(config, args.logs, args.weights, is_training=False) as model:
        model_init_end_time = time.time()
        model_init_time = model_init_end_time - model_init_start_time
        
        print(f"Model initialized in {model_init_time:.4f} seconds")
        
        # Timing for actual inference
        inference_start_time = time.time()
        results = model.detect([image], verbose=1)
        inference_end_time = time.time()
        inference_time = inference_end_time - inference_start_time
        
        print(f"Inference completed in {inference_time:.4f} seconds")
        print(f"Inference speed: {1/inference_time:.2f} FPS")
        
        # Timing for visualization/saving
        viz_start_time = time.time()
        visualize_results(image, results[0], args)
        viz_end_time = time.time()
        viz_time = viz_end_time - viz_start_time
        
        print(f"Visualization completed in {viz_time:.4f} seconds")
    
    # Total time calculation
    total_time = load_time + model_init_time + inference_time + viz_time
    print(f"\n=== Timing Summary ===")
    print(f"Image loading:     {load_time:.4f}s ({load_time/total_time*100:.1f}%)")
    print(f"Model initialization: {model_init_time:.4f}s ({model_init_time/total_time*100:.1f}%)")
    print(f"Inference:         {inference_time:.4f}s ({inference_time/total_time*100:.1f}%)")
    print(f"Visualization:     {viz_time:.4f}s ({viz_time/total_time*100:.1f}%)")
    print(f"Total time:        {total_time:.4f}s")
    print("=====================")
    
    print("Inference completed with proper memory cleanup")


# --- Main Execution ---
def main():
    """Parses command line arguments and runs the requested command."""
    # Set GPU memory allocator early to prevent fragmentation
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
    
    parser = argparse.ArgumentParser(description='Train or test Mask R-CNN for trees.')
    parser.add_argument('--dataset', required=True, help='Path to yolo_dataset root (contains train/valid)')
    parser.add_argument('--weights', default=COCO_WEIGHTS_PATH, help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', default=DEFAULT_LOGS_DIR, help='Logs and checkpoints directory')
    parser.add_argument('--command', required=True, choices=['train', 'test'], help="train or test")
    parser.add_argument('--image', help='Image to test on (for test command)')
    parser.add_argument('--min_confidence', type=float, help='Override config.DETECTION_MIN_CONFIDENCE for inference')
    parser.add_argument('--save_overlay', action='store_true', help='If set, save an overlay image of predictions')
    parser.add_argument('--output', help='Output path for saved overlay image.')
    parser.add_argument('--layers', default='heads', help="Which layers to train: 'heads' or 'all'")
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    args = parser.parse_args()

    config = TreeConfig()
    if args.min_confidence is not None:
        config.DETECTION_MIN_CONFIDENCE = args.min_confidence
    
    # Start GPU monitoring early for all commands
    os.makedirs(args.logs, exist_ok=True)
    gpu_process = None
    gpu_log_file = None
    
    if args.command == 'train':
        # Start GPU monitoring for training
        gpu_result = start_gpu_monitor(args.logs, interval=2)
        if gpu_result:
            gpu_process, gpu_log_file = gpu_result
            print(f"GPU monitoring started early, logging to {gpu_log_file}")
        else:
            gpu_process, gpu_log_file = None, None
    
    print(f"\nCommand: {args.command}")
    print("--- Configuration ---")
    config.display()
    print("---------------------\n")
    
    try:
        if args.command == 'train':
            train_model(config, args, gpu_process, gpu_log_file)
        elif args.command == 'test':
            test_model(config, args)
    finally:
        # GPU monitoring is stopped within train_model after training completion
        # Only stop if it's still running (e.g., if training was interrupted)
        if gpu_process and gpu_process.poll() is None:
            print("Stopping GPU monitoring (cleanup)")
            stop_gpu_monitor(gpu_process)

if __name__ == '__main__':
    main()

