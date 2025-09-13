import os
import cv2
import torch
import logging
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from sklearn.model_selection import train_test_split
import argparse

# --- Argument parser ---
parser = argparse.ArgumentParser(description="Train UNet for Shadow-Casting Object Segmentation")
parser.add_argument("--image_dir", type=str, default="dataset/unet_dataset/images", help="Path to input images")
parser.add_argument("--mask_dir", type=str, default="dataset/unet_dataset/masks/train", help="Path to masks")
parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save models and plots")
parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
parser.add_argument("--num_epochs", type=int, default=20, help="Number of training epochs")
args = parser.parse_args()

IMAGE_DIR = args.image_dir
MASK_DIR = args.mask_dir
OUTPUT_DIR = args.output_dir
BATCH_SIZE = args.batch_size
NUM_EPOCHS = args.num_epochs

# --- Config ---
LR = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ENCODER = 'resnet34'
CLASSES = 2
IMG_SIZE = (512, 512)

# Create directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "models"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "plots"), exist_ok=True)

# Logging
logging.basicConfig(level=logging.INFO)

# --- Dataset ---
class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, size=(512, 512)):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.size = size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.size)
        image = image / 255.0
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, self.size, interpolation=cv2.INTER_NEAREST)
        mask = mask.astype(np.int64)

        return torch.tensor(image), torch.tensor(mask)

# --- File paths ---
image_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith(('.jpg', '.png'))])
mask_files = sorted([f for f in os.listdir(MASK_DIR) if f.endswith('.png')])

image_paths = [os.path.join(IMAGE_DIR, f) for f in image_files]
mask_paths = [os.path.join(MASK_DIR, f) for f in mask_files]

train_imgs, val_imgs, train_masks, val_masks = train_test_split(image_paths, mask_paths, test_size=0.2, random_state=42)

train_ds = SegmentationDataset(train_imgs, train_masks, IMG_SIZE)
val_ds = SegmentationDataset(val_imgs, val_masks, IMG_SIZE)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=1)

# --- Model ---
model = smp.Unet(
    encoder_name=ENCODER,
    encoder_weights='imagenet',
    in_channels=3,
    classes=CLASSES
).to(DEVICE)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# --- Training ---
def train_epoch(loader, model, optimizer, loss_fn):
    model.train()
    epoch_loss = 0
    for images, masks in tqdm(loader):
        images, masks = images.to(DEVICE), masks.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, masks)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(loader)

# --- Evaluation (IoU + F1) ---
def evaluate(loader, model, num_classes=2):
    model.eval()
    ious, f1s = [], []
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            for cls in range(1, num_classes):
                tp = ((preds == cls) & (masks == cls)).sum().item()
                fp = ((preds == cls) & (masks != cls)).sum().item()
                fn = ((preds != cls) & (masks == cls)).sum().item()
                union = ((preds == cls) | (masks == cls)).sum().item()

                iou = tp / union if union > 0 else 1.0
                precision = tp / (tp + fp + 1e-7)
                recall = tp / (tp + fn + 1e-7)
                f1 = 2 * precision * recall / (precision + recall + 1e-7)

                ious.append(iou)
                f1s.append(f1)
    return np.mean(ious), np.mean(f1s)

# --- Training Loop ---
for epoch in range(NUM_EPOCHS):
    logging.info(f"Epoch {epoch+1}/{NUM_EPOCHS}")
    train_loss = train_epoch(train_loader, model, optimizer, loss_fn)
    val_iou, val_f1 = evaluate(val_loader, model)
    logging.info(f"Train Loss: {train_loss:.4f} | Val IoU: {val_iou:.4f} | Val F1: {val_f1:.4f}")

# --- Save Model ---
torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "models", "unet_model.pth"))

# --- Save Training Metrics ---
plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.plot(val_ious, label="Val IoU")
plt.plot(val_f1s, label="Val F1")
plt.xlabel("Epochs")
plt.title("Training Metrics")
plt.legend()
plt.savefig(os.path.join(OUTPUT_DIR, "plots", "training_metrics.png"))
plt.close()