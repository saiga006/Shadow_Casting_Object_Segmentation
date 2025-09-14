# Tree Instance Segmentation with Mask R-CNN
# Adapts balloon.py for aerial tree segmentation
import os
import sys
import numpy as np
import skimage.io
import yaml
import cv2

# Optional: limit GPU memory so training/inference don't grab all GPU RAM
try:
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Limit to 4GB (adjust based on your needs)
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
            )
            print("GPU memory limit set to 4GB")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(f"GPU configuration error: {e}")
except Exception:
    # TensorFlow may not be available in all environments; continue without GPU limiting
    pass

from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "../mask_rcnn_coco.h5")
DEFAULT_LOGS_DIR = os.path.join(os.path.dirname(__file__), "../logs")

class TreeConfig(Config):
    NAME = "tree"
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1  # background + tree
    STEPS_PER_EPOCH = 200
    # Preserve aspect ratio and pad to multiple of 64 (good for small fixed-size images)
    IMAGE_RESIZE_MODE = "pad64"
    # Image size for this project (your images are ~500x500).
    # Must be divisible by 2^6 (64). Use 512 to satisfy model down/up-scaling requirements.
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    # Anchor scales tuned for trees
    RPN_ANCHOR_SCALES = (16, 32, 64, 128)
    # Lower default confidence during debugging; raise later for final evaluation
    DETECTION_MIN_CONFIDENCE = 0.80
    # Ensure a sensible maximum number of instances and NMS for this dataset
    DETECTION_MAX_INSTANCES = 150
    # NMS IOU
    DETECTION_NMS_THRESHOLD = 0.3
    # ROIs used for training the classifier/mask head
    ROI_POSITIVE_RATIO = 0.4
    # ROIs used for training the classifier/mask head
    TRAIN_ROIS_PER_IMAGE = 50

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
        image_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg") or f.endswith(".png")]
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
        masks = []
        class_ids = []
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 7:
                    continue
                # YOLO format: class_id x1 y1 x2 y2 ... (normalized)
                coords = [float(x) for x in parts[1:]]
                if len(coords) % 2 != 0:
                    continue
                poly = np.array(coords).reshape(-1, 2)
                poly[:, 0] *= width
                poly[:, 1] *= height
                poly = poly.astype(np.int32)
                mask = np.zeros((height, width), dtype=np.uint8)
                if len(poly) >= 3:
                    cv2.fillPoly(mask, [poly], color=1)
                    masks.append(mask)
                    class_ids.append(1)  # tree
        if masks:
            mask = np.stack(masks, axis=-1)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            return np.empty((height, width, 0)), np.array([], dtype=np.int32)

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info["path"]

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train or test Mask R-CNN for trees.')
    parser.add_argument('--dataset', required=True, help='Path to yolo_dataset root (contains train/valid)')
    parser.add_argument('--weights', required=False, default=COCO_WEIGHTS_PATH, help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False, default=DEFAULT_LOGS_DIR, help='Logs and checkpoints directory')
    parser.add_argument('--command', required=True, choices=['train', 'test'], help="train or test")
    parser.add_argument('--image', required=False, help='Image to test on (for test command)')
    parser.add_argument('--min_confidence', required=False, type=float, default=None,
                        help='Override config.DETECTION_MIN_CONFIDENCE for inference (e.g. 0.3)')
    parser.add_argument('--save_overlay', action='store_true',
                        help='If set, save an overlay image of predictions instead of showing display_instances')
    parser.add_argument('--output', required=False, default=None,
                        help='Output path for saved overlay image. If not set and --save_overlay is used, saves next to input image.')
    parser.add_argument('--layers', required=False, default='heads',
                        help="Which layers to train: 'heads' or 'all' (default: 'heads')")
    args = parser.parse_args()

    config = TreeConfig()
    if args.min_confidence is not None:
        config.DETECTION_MIN_CONFIDENCE = float(args.min_confidence)

    if args.command == 'train':
        print('\n=== Active configuration (training) ===')
        config.display()
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

        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=30,
                    layers=args.layers)

    elif args.command == 'test':
        print('\n=== Active configuration (inference) ===')
        config.display()
        model = modellib.MaskRCNN(mode='inference', config=config, model_dir=args.logs)
        model.load_weights(args.weights, by_name=True)
        assert args.image, 'Provide --image for test command'
        image = skimage.io.imread(args.image)
        results = model.detect([image], verbose=1)
        r = results[0]

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

            out_path = args.output
            if not out_path:
                base, _ = os.path.splitext(os.path.basename(args.image))
                out_path = os.path.join(os.getcwd(), f"{base}_pred_overlay.jpg")
            cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            print(f"Saved prediction overlay to: {out_path}")

        if not args.save_overlay:
            import matplotlib.pyplot as plt
            from mrcnn.visualize import display_instances
            display_instances(image, r['rois'], r['masks'], r['class_ids'], ['BG', 'tree'], r['scores'])
            plt.show()
