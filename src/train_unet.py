import os
import cv2
import torch
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

BATCH_SIZE = 4
NUM_EPOCHS = 20
LR = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ENCODER = 'resnet34'
CLASSES = 2
IMG_SIZE = (512, 512)

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

# --- Training ---
def train_epoch(loader, model, optimizer, loss_fn):
    model.train()
    epoch_loss = 0
    for images, masks in tqdm(loader):
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, masks)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(loader)

# --- Validation ---
def evaluate(loader, model):
    model.eval()
    ious = []
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            intersection = ((preds == 1) & (masks == 1)).sum().item()
            union = ((preds == 1) | (masks == 1)).sum().item()
            if union == 0:
                iou = 1.0
            else:
                iou = intersection / union
            ious.append(iou)
    return np.mean(ious)

# --- Loop ---
train_losses = []
val_ious = []

for epoch in range(NUM_EPOCHS):
    logging.info(f"Epoch {epoch+1}/{NUM_EPOCHS}")
    train_loss = train_epoch(train_loader, model, optimizer, loss_fn)
    val_iou = evaluate(val_loader, model)

    train_losses.append(train_loss)
    val_ious.append(val_iou)

    logging.info(f"Train Loss: {train_loss:.4f} | Val IoU: {val_iou:.4f}")

# --- Save Model ---

torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "models", "unet_model.pth"))


# --- Plot ---
plt.plot(train_losses, label='Train Loss')
plt.plot(val_ious, label='Val IoU')
plt.legend()
plt.title('Training Curve')
plt.savefig(os.path.join(OUTPUT_DIR, "plots", "training_plot.png"))
