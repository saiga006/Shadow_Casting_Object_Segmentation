import os
import cv2
import numpy as np
from tqdm import tqdm

# Paths (update as needed)
img_dir = '../dataset/yolo_dataset/train/images'
label_dir = '../dataset/yolo_dataset/train/labels'
mask_dir = '../dataset/unet_dataset/masks/train'

os.makedirs(mask_dir, exist_ok=True)

# Helper to convert normalized coords to absolute pixel values
def denormalize_coords(coords, width, height):
    coords = np.array(coords).reshape(-1, 2)
    coords[:, 0] *= width
    coords[:, 1] *= height
    return coords.astype(np.int32)

for label_file in tqdm(os.listdir(label_dir)):
    if not label_file.endswith('.txt'):
        continue

    base_name = os.path.splitext(label_file)[0]
    img_path = os.path.join(img_dir, base_name + '.jpg')
    label_path = os.path.join(label_dir, label_file)

    if not os.path.exists(img_path):
        print(f"Image not found for {label_file}")
        continue

    img = cv2.imread(img_path)
    height, width = img.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)

    with open(label_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            class_id = int(parts[0])  # 0 = shadow-casting object
            coords = list(map(float, parts[1:]))

            if len(coords) % 2 != 0:
                print(f"Invalid coords in {label_file}: {coords}")
                continue

            poly = denormalize_coords(coords, width, height)
            cv2.fillPoly(mask, [poly], color=class_id + 1)  # 1 = object

    cv2.imwrite(os.path.join(mask_dir, base_name + '.png'), mask)
