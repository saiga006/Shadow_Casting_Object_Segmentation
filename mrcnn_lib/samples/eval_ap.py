"""
Evaluate Mask R-CNN on a YOLO-style dataset using the TreeDataset and TreeConfig.
Computes per-image AP (using mrcnn.utils.compute_ap) and writes a CSV summary.

Usage:
    python3 samples/eval_ap.py --dataset /path/to/yolo_dataset --subset valid \
        --weights /path/to/weights.h5 --output results.csv --iou 0.5 --limit 100 --save_overlays outdir

"""
import os
import sys
import csv
import argparse
import numpy as np
import skimage.io
import cv2

# Ensure project root is on sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from mrcnn import model as modellib
from mrcnn import utils
# Import local config/dataset
from samples.tree_segmentation import TreeConfig, TreeDataset


def save_overlay(image, gt_masks, pred_masks, out_path, alpha=0.4):
    overlay = image.copy()
    # draw GT masks in blue
    if gt_masks is not None and gt_masks.size > 0:
        for i in range(gt_masks.shape[-1]):
            m = gt_masks[..., i].astype(bool)
            colored = np.zeros_like(image, dtype=np.uint8)
            colored[m] = (0, 0, 255)  # red channel BGR later, but image is RGB here
            overlay = np.where(m[..., None], (overlay * (1 - alpha) + colored * alpha).astype(np.uint8), overlay)
    # draw predicted masks in green
    if pred_masks is not None and pred_masks.size > 0:
        for i in range(pred_masks.shape[-1]):
            m = pred_masks[..., i].astype(bool)
            colored = np.zeros_like(image, dtype=np.uint8)
            colored[m] = (0, 255, 0)
            overlay = np.where(m[..., None], (overlay * (1 - alpha) + colored * alpha).astype(np.uint8), overlay)
    # draw contours
    if pred_masks is not None and pred_masks.size > 0:
        for i in range(pred_masks.shape[-1]):
            m = (pred_masks[..., i] * 255).astype(np.uint8)
            contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (255, 255, 255), 1)
    # Convert RGB->BGR for saving with OpenCV
    cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


def evaluate(dataset_dir, subset, weights_path, logs_dir, limit=None, iou_threshold=0.5, output_csv=None, save_overlays=None, min_confidence=None):
    # Prepare dataset
    dataset = TreeDataset()
    dataset.load_tree(dataset_dir, subset)
    dataset.prepare()

    # Config
    config = TreeConfig()
    config.GPU_COUNT = 1
    config.IMAGES_PER_GPU = 1
    if min_confidence is not None:
        config.DETECTION_MIN_CONFIDENCE = float(min_confidence)

    # Model
    model = modellib.MaskRCNN(mode="inference", config=config, model_dir=logs_dir)
    model.load_weights(weights_path, by_name=True)

    image_ids = dataset.image_ids
    if limit:
        image_ids = image_ids[:limit]

    results = []

    for image_id in image_ids:
        info = dataset.image_info[image_id]
        image = skimage.io.imread(info['path'])
        # GT
        gt_masks, gt_class_ids = dataset.load_mask(image_id)
        # convert gt_masks to boolean
        if gt_masks.size == 0:
            gt_masks_arr = np.empty((image.shape[0], image.shape[1], 0), dtype=bool)
            gt_class_ids = np.array([], dtype=np.int32)
        else:
            gt_masks_arr = (gt_masks > 0).astype(np.bool_)
        # Predict
        r = model.detect([image], verbose=0)[0]
        pred_masks = r.get('masks')
        pred_class_ids = r.get('class_ids')
        pred_scores = r.get('scores')
        pred_boxes = r.get('rois')

        # Debug: log raw predictions for diagnosis
        print(f"DEBUG: preds -> scores:{pred_scores} class_ids:{pred_class_ids} boxes:{pred_boxes}")

        # Compute gt_boxes from masks
        if gt_masks_arr.size == 0:
            gt_boxes = np.zeros((0, 4), dtype=np.int32)
        else:
            gt_boxes = utils.extract_bboxes(gt_masks_arr)

        # Ensure pred arrays exist
        if pred_masks is None or pred_masks.size == 0:
            pred_masks = np.empty((image.shape[0], image.shape[1], 0), dtype=np.bool_)
            pred_boxes = np.zeros((0, 4), dtype=np.int32)
            pred_class_ids = np.array([], dtype=np.int32)
            pred_scores = np.array([], dtype=np.float32)

        # Compute AP for this image
        try:
            ap, precisions, recalls, overlaps = utils.compute_ap(
                gt_boxes, gt_class_ids, gt_masks_arr,
                pred_boxes, pred_class_ids, pred_scores, pred_masks,
                iou_threshold=iou_threshold)
        except Exception as e:
            # Fallback: compute overlaps and simple matching if compute_ap not available
            print(f"compute_ap failed for image {info['path']}: {e}")
            ap = 0.0

        results.append({
            'image_id': info.get('id', image_id),
            'path': info['path'],
            'num_gt': gt_masks_arr.shape[-1],
            'num_pred': pred_masks.shape[-1],
            'ap': float(ap)
        })

        # Save overlay if requested
        if save_overlays:
            base = os.path.splitext(os.path.basename(info['path']))[0]
            out_path = os.path.join(save_overlays, f"{base}_eval_overlay.jpg")
            os.makedirs(save_overlays, exist_ok=True)
            save_overlay(image, gt_masks_arr.astype(np.uint8), pred_masks.astype(np.uint8), out_path)

        print(f"{info['path']}: GT={results[-1]['num_gt']} PRED={results[-1]['num_pred']} AP={results[-1]['ap']:.3f}")

    # Write CSV
    if output_csv:
        keys = ['image_id', 'path', 'num_gt', 'num_pred', 'ap']
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for row in results:
                writer.writerow(row)
        print(f"Saved results to {output_csv}")

    # Summary
    ap_values = [r['ap'] for r in results]
    mean_ap = float(np.mean(ap_values)) if ap_values else 0.0
    print(f"Mean AP (@{iou_threshold}): {mean_ap:.4f}")
    return mean_ap, results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Mask R-CNN on a dataset subset and compute AP per image.')
    parser.add_argument('--dataset', required=True, help='Path to yolo_dataset root (contains train/valid)')
    parser.add_argument('--subset', required=False, default='valid', help='Which subset to evaluate: train or valid')
    parser.add_argument('--weights', required=True, help='Path to weights .h5 file')
    parser.add_argument('--logs', required=False, default=os.path.join(ROOT_DIR, 'logs'), help='Logs directory')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of images to evaluate')
    parser.add_argument('--iou', type=float, default=0.5, help='IoU threshold for AP')
    parser.add_argument('--output', required=False, default='eval_results.csv', help='Output CSV file')
    parser.add_argument('--save_overlays', required=False, default=None, help='Directory to save overlays (optional)')
    parser.add_argument('--min_confidence', required=False, default=None, help='Override detection min confidence')
    args = parser.parse_args()

    evaluate(args.dataset, args.subset, args.weights, args.logs, limit=args.limit, iou_threshold=args.iou, output_csv=args.output, save_overlays=args.save_overlays, min_confidence=args.min_confidence)
