# Shadow Casting Object Segmentation 🌑🖼️

## Overview 🎯

This repository contains code and datasets related to **shadow casting object segmentation**. The project focuses on detecting and segmenting objects in images affected by shadows, a challenging task in computer vision due to the complex interaction between objects and shadow patterns.

The dataset included follows the YOLO format for object detection training. The segmentation approach aims to improve accuracy in scenarios with significant shadow effects.



## Features ✨

- 📂 Dataset prepared in YOLO format for shadow-affected object segmentation.
- 🛠️ Scripts and models for training and inference.
- 🔧 Tools for preprocessing and annotation support.

## Installation ⚙️

1. Clone the repository:

```bash
git clone https://github.com/ItsShriks/Shadow_Casting_Object_Segmentation.git
cd Shadow_Casting_Object_Segmentation
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

### Training 🏋️‍♂️

```bash
python train.py --data dataset/yolo_dataset --epochs 50 --batch-size 16
```
### Inference 🔍
```bash
python inference.py --weights best_model.pth --image path/to/image.jpg
```

---

## Contributing 🤝

Feel free to open issues or submit pull requests for improvements and bug fixes.

---

## Authors ✍️

- [Shrikar Nakhye](https://www.linkedin.com/in/shrikar-n-053262188/) – [📧 Email](mailto:shrikar.nakhye@smail.inf.h-brs.de)

- [Sai Mukkundan](mailto:sai.ramamoorthy@smail.inf.h-brs.de) – [📧 Email](mailto:sai.ramamoorthy@smail.inf.h-brs.de)
- [Kai Glasenapp](mailto:kai.glasenapp@smail.inf.h-brs.de) – [📧 Email](mailto:kai.glasenapp@smail.inf.h-brs.de)

---

## Acknowledgments 🙏

This project was developed as part of the coursework for the DLRV – Deep Learning for Robot Vision class at Hochschule Bonn-Rhein-Sieg during Summer Semester 2025.

Special thanks to:

- [**Prof. Dr. Sebastian Houben**](sebastian.houben@h-brs.de)
  For his guidance, valuable insights, and continuous support throughout the course and project.
