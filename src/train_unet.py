import os
import cv2
import torch
import time
import logging
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# --- Config ---
IMAGE_DIR = os.path.join("dataset", "unet_dataset", "images")
MASK_DIR = os.path.join("dataset", "unet_dataset", "masks", "train")
OUTPUT_DIR = os.path.join("outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "models"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "plots"), exist_ok=True)

BATCH_SIZE = 4
NUM_EPOCHS = 1
LR = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ENCODER = 'resnet34'
CLASSES = 2
IMG_SIZE = (512, 512)

# Logging
log_dir = os.path.join(OUTPUT_DIR, "logs")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(log_dir, "training.log"),
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)

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

# --- Collect file paths ---
image_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith(('.jpg', '.png'))])
mask_files = sorted([f for f in os.listdir(MASK_DIR) if f.endswith('.png')])

image_paths = [os.path.join(IMAGE_DIR, f) for f in image_files]
mask_paths = [os.path.join(MASK_DIR, f) for f in mask_files]

# --- Split ---
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

# --- GPU Memory Logger ---
def log_gpu_memory():
    if torch.cuda.is_available():
        mem_alloc = torch.cuda.memory_allocated() / (1024 ** 2)
        mem_reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        logging.info(f"GPU Memory Allocated: {mem_alloc:.2f} MB | Reserved: {mem_reserved:.2f} MB")

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

            for cls in range(1, num_classes):  # skip background (0)
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

# --- Inference Time ---
def measure_inference_time(model, loader, device, n_warmup=5):
    model.eval()
    times = []
    with torch.no_grad():
        # Warm-up
        for _ in range(n_warmup):
            for images, _ in loader:
                images = images.to(device)
                _ = model(images)
                break
        # Actual timing
        for images, _ in loader:
            images = images.to(device)
            start = time.time()
            _ = model(images)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end = time.time()
            times.append(end - start)
    return np.mean(times), np.std(times)

# --- Loop ---
train_losses, val_ious, val_f1s, epoch_times = [], [], [], []

for epoch in range(NUM_EPOCHS):
    start_time = time.time()
    logging.info(f"Epoch {epoch+1}/{NUM_EPOCHS}")

    train_loss = train_epoch(train_loader, model, optimizer, loss_fn)
    val_iou, val_f1 = evaluate(val_loader, model)
    end_time = time.time()
    epoch_time = end_time - start_time

    log_gpu_memory()

    train_losses.append(train_loss)
    val_ious.append(val_iou)
    val_f1s.append(val_f1)
    epoch_times.append(epoch_time)

    logging.info(f"Train Loss: {train_loss:.4f} | Val IoU: {val_iou:.4f} | Val F1: {val_f1:.4f} | Time: {epoch_time:.2f}s")

# --- Save Model ---
torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "models", "unet_model.pth"))

# --- Inference Speed ---
avg_inf, std_inf = measure_inference_time(model, val_loader, DEVICE)
logging.info(f"Inference time per batch: {avg_inf:.4f} ± {std_inf:.4f} sec")

# --- Plots ---
plt.figure()
plt.plot(train_losses, label='Train Loss')
plt.plot(val_ious, label='Val IoU')
plt.plot(val_f1s, label='Val F1')
plt.legend()
plt.title('Training Metrics')
plt.savefig(os.path.join(OUTPUT_DIR, "plots", "metrics_plot.png"))
