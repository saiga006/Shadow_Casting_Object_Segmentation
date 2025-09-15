# 🌍 PrithviVision

## Overview 🎯
**PrithviVision** is a deep learning project for **shadow-casting object segmentation in aerial imagery**.
The goal is to detect and segment objects affected by shadows — a common challenge in remote sensing and urban scene understanding.

The project combines **U-Net, YOLO, and Mask R-CNN** approaches to handle both **object detection** and **pixel-level segmentation**. The dataset is curated from **aerial images of Bonn city**, annotated in **YOLO format**.

---

## Features ✨
- 📂 **YOLO-format dataset** prepared for aerial shadow segmentation.
- 🧠 **Multiple models supported** – U-Net, YOLO, Mask R-CNN.
- 🛠️ **Preprocessing & annotation utilities** for dataset preparation.
- ⚡ Modular training and inference scripts.

---

## Installation ⚙️

1. Clone the repository:

```bash
git clone https://github.com/ItsShriks/Shadow_Casting_Object_Segmentation.git
cd PrithviVision
```
2. Create a conda environment with required dependencies:
```bash
conda env create -f Essentials/dlrv.yml
```
3. Activate the environment:
```bash
conda activate dlrv
```
---

## Usage 🚀

### Dataset 📁
 The dataset/yolo_dataset/ directory contains the dataset annotated in YOLO format.

Images and their corresponding label files are organized for training.

<!--### Training YOLO🏋️‍♂️

```bash
python train.py --data dataset/yolo_dataset --epochs 50 --batch-size 16
```-->
<!--### Inference 🔍
```bash
python inference.py --weights best_model.pth --image path/to/image.jpg
```-->

### Training UNet 🏋️‍♂️
```bash
python src/train_unet.py
```
🧪 Notes for U-Net
- U-Net uses 2 classes: background (0) and shadow-casting object (1).
- Input images and masks are resized to 512×512 during training.
### Inference 🔍
```bash
python inference_unet.py
```

<!--### Training Mask R-CNN (Optional) 🖼️-->

---

## Contributing 🤝

Contributions are welcome!
Open issues or submit pull requests to improve PrithviVision.

---

## Authors ✍️

- [Shrikar Nakhye](https://www.linkedin.com/in/shrikar-n-053262188/) – [📧 Email](mailto:shrikar.nakhye@smail.inf.h-brs.de)

- [Sai Mukkundan](mailto:sai.ramamoorthy@smail.inf.h-brs.de) – [📧 Email](mailto:sai.ramamoorthy@smail.inf.h-brs.de)

---

## Acknowledgments 🙏

This project was developed as part of the coursework for the DLRV – Deep Learning for Robot Vision class at Hochschule Bonn-Rhein-Sieg during Summer Semester 2025.

Special thanks to:

- [**Prof. Dr. Sebastian Houben**](sebastian.houben@h-brs.de)
  For his guidance, valuable insights, and continuous support throughout the course and project.
