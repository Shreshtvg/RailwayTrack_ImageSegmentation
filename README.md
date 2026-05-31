# Railway Track Semantic Segmentation

## Overview

This project performs semantic segmentation of railway tracks from drone imagery using a U-Net architecture with a pretrained ResNet34 encoder.

The model takes a railway drone image as input and generates:

* Binary segmentation mask
* Railway overlay visualization
* Pixel-level railway extraction

---

## Project Workflow

```text
Drone Image
      ↓
Preprocessing
      ↓
U-Net + ResNet34
      ↓
Segmentation Mask
      ↓
Overlay Visualization
```

---

## Dataset

Dataset Split:

| Dataset    | Images |
| ---------- | ------ |
| Train      | 297    |
| Validation | 27     |
| Test       | 18     |

The dataset contains drone-captured railway images with corresponding pixel-level segmentation masks.

Mask Format:

* White Pixels → Railway Track
* Black Pixels → Background

---

## Model Architecture

### Segmentation Model

* Architecture: U-Net
* Encoder: ResNet34
* Framework: PyTorch
* Encoder Weights: ImageNet Pretrained

### Why U-Net?

U-Net is specifically designed for semantic segmentation tasks and performs well on relatively small datasets while maintaining accurate pixel-level localization.

---

## Training Configuration

| Parameter     | Value                |
| ------------- | -------------------- |
| Image Size    | 512 × 512            |
| Batch Size    | 2                    |
| Epochs        | 50                   |
| Learning Rate | 1e-4                 |
| Optimizer     | AdamW                |
| Loss Function | Dice Loss + BCE Loss |

---

## Evaluation Results

| Metric          | Score  |
| --------------- | ------ |
| Dice Score      | 90.33% |
| IoU Score       | 87.98% |
| Validation Loss | 0.1385 |

These results indicate strong railway segmentation performance on unseen test images.

---

## Requirements

Python Version:

```text
Python 3.11
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Project Structure

```text
RailwaySegmentation/

├── outputs/
│   └── checkpoints/
│       └── railway_unet_resnet34_best.pth

├── model.py
├── predict_segmentation.py
├── requirements.txt
└── README.md
```

---

## Running Inference

### 1. Update Image Path

Inside:

```text
predict_segmentation.py
```

Set:

```python
IMAGE_PATH = "path_to_image.jpg"
```

---

### 2. Run Prediction

```bash
python predict_segmentation.py
```

---

### 3. Outputs

The script generates:

* Predicted Segmentation Mask
* Overlay Visualization

Output directories:

```text
outputs/masks/
outputs/overlays/
```

---

## Output Visualization

The prediction interface displays:

* Original Image
* Predicted Segmentation Mask
* Overlay Visualization

This allows easy comparison between model predictions and annotated masks.

---

## Hardware Used

* GPU: NVIDIA RTX 3050 Laptop GPU
* CUDA Enabled Training
* PyTorch 2.11.0 + CUDA 12.8

---

## Future Enhancements

Planned next steps:

* Additional dataset expansion
* Model fine-tuning
* Skeletonization
* Railway centerline extraction
* Railway geometry analysis
* Discontinuity and defect detection
* Web-based inference dashboard

---

## Author

Shresht V G
