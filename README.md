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

## Training the Model

### Dataset Structure

Organize the dataset in the following format:

```text
dataset/

├── train/
│   ├── images/
│   └── masks/
│
├── valid/
│   ├── images/
│   └── masks/
│
└── test/
    ├── images/
    └── masks/
```

Where:

* Images are RGB railway drone images.
* Masks are binary segmentation masks.
* White pixels (255) represent railway tracks.
* Black pixels (0) represent background.

---

### Configure Training Parameters

Update values inside:

```text
config.py
```

Example:

```python
IMAGE_SIZE = 512
BATCH_SIZE = 2
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
```

---

### Start Training

Run:

```bash
python train.py
```

The training pipeline will:

1. Load images and masks.
2. Apply preprocessing and augmentation.
3. Train the U-Net + ResNet34 segmentation model.
4. Validate performance after each epoch.
5. Save the best-performing model checkpoint.

---

### Checkpoint Location

The best model is automatically saved to:

```text
outputs/checkpoints/
```

Example:

```text
outputs/checkpoints/railway_unet_resnet34.pth
```

---

### Training Outputs

During training the following metrics are displayed:

* Training Loss
* Validation Loss
* Dice Score
* IoU Score

Example:

```text
Epoch      : 50
Train Loss : 0.0705
Val Loss   : 0.1385
Dice Score : 0.9033
IoU Score  : 0.8798
```

---

### Custom Dataset Training

To train on a new railway dataset:

1. Replace images and masks inside the dataset folders.
2. Maintain the same folder structure.
3. Update paths in `config.py` if required.
4. Run:

```bash
python train.py
```

The model will automatically learn from the new dataset and generate a new checkpoint.

---

## Author

Shresht V G
