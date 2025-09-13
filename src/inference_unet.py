import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import argparse

# --- Config ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = (512, 512)
CLASSES = 2
ENCODER = 'resnet34'

# --- Inference function ---
def infer_image(model, image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, IMG_SIZE)
    image_norm = image_resized / 255.0
    input_tensor = torch.tensor(np.transpose(image_norm, (2,0,1)), dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(input_tensor)
        pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()

    return image_rgb, pred_mask

# --- Visualize ---
def visualize(image, mask):
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.imshow(image)
    plt.title("Input Image")
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.imshow(mask, cmap='gray')
    plt.title("Predicted Mask")
    plt.axis('off')
    plt.show()

# --- Main ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UNet Inference for Shadow-Casting Objects")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image")
    parser.add_argument("--weights", type=str, required=True, help="Path to the trained UNet model (.pth)")
    args = parser.parse_args()

    # Load model
    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=None,
        in_channels=3,
        classes=CLASSES
    ).to(DEVICE)
    model.load_state_dict(torch.load(args.weights, map_location=DEVICE))
    model.eval()

    # Run inference
    img, mask = infer_image(model, args.image)
    visualize(img, mask)
