import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

# ==========================================
# CONFIG
# ==========================================

MODEL_PATH = "outputs/checkpoints/railway_unet_resnet34_best.pth"

IMAGE_PATH = "dataset/test/images/DJI_20211217135551_0055_JPG_jpg.rf.e1d479c7123389d6a09723856539ab2c.jpg"

# MASK_PATH = "dataset/test/masks/DJI_20211217135424_0001_JPG_jpg.rf.e9389e07eee7477f2a978fc2a9c33327.png"

OUTPUT_MASK = "outputs/masks/prediction_mask.png"

OUTPUT_OVERLAY = "outputs/overlays/prediction_overlay.png"

IMAGE_SIZE = 512

# ==========================================
# CREATE OUTPUT FOLDERS
# ==========================================

os.makedirs(
    "outputs/masks",
    exist_ok=True
)

os.makedirs(
    "outputs/overlays",
    exist_ok=True
)

# ==========================================
# DEVICE
# ==========================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Using Device:", DEVICE)

# ==========================================
# LOAD MODEL
# ==========================================

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=1
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.to(DEVICE)

model.eval()

print("Model Loaded")

# ==========================================
# READ IMAGE
# ==========================================

image = cv2.imread(IMAGE_PATH)

original = image.copy()

h, w = image.shape[:2]

image_rgb = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)

# ==========================================
# PREPROCESS
# ==========================================

resized = cv2.resize(
    image_rgb,
    (IMAGE_SIZE, IMAGE_SIZE)
)

resized = resized.astype(
    np.float32
) / 255.0

tensor = torch.tensor(
    resized.transpose(2, 0, 1),
    dtype=torch.float32
).unsqueeze(0)

tensor = tensor.to(DEVICE)

# ==========================================
# PREDICTION
# ==========================================

with torch.no_grad():

    output = model(tensor)

    output = torch.sigmoid(output)

    output = output.squeeze().cpu().numpy()

# ==========================================
# THRESHOLD
# ==========================================

pred_mask = (
    output > 0.5
).astype(np.uint8) * 255

# ==========================================
# RESIZE BACK
# ==========================================

pred_mask = cv2.resize(
    pred_mask,
    (w, h),
    interpolation=cv2.INTER_NEAREST
)

# ==========================================
# SAVE MASK
# ==========================================

cv2.imwrite(
    OUTPUT_MASK,
    pred_mask
)

print("Mask Saved")

# ==========================================
# CREATE OVERLAY
# ==========================================

overlay = original.copy()

overlay[pred_mask > 0] = [0, 0, 255]

result = cv2.addWeighted(
    original,
    0.7,
    overlay,
    0.3,
    0
)

cv2.imwrite(
    OUTPUT_OVERLAY,
    result
)

print("Overlay Saved")

# ==========================================
# LOAD GROUND TRUTH
# ==========================================

# gt_mask = cv2.imread(
#     MASK_PATH,
#     cv2.IMREAD_GRAYSCALE
# )

# ==========================================
# DISPLAY RESULTS
# ==========================================

plt.figure(figsize=(22, 6))

# Original
plt.subplot(1, 4, 1)

plt.imshow(
    cv2.cvtColor(
        original,
        cv2.COLOR_BGR2RGB
    )
)

plt.title("Original")

plt.axis("off")

# Ground Truth
# plt.subplot(1, 4, 2)

# plt.imshow(
#     gt_mask,
#     cmap="gray"
# )

# plt.title("Mask generated for training")

# plt.axis("off")

# Prediction
plt.subplot(1, 4, 2)

plt.imshow(
    pred_mask,
    cmap="gray"
)

plt.title("Mask Prediction after training")

plt.axis("off")

# Overlay
plt.subplot(1, 4, 3)

plt.imshow(
    cv2.cvtColor(
        result,
        cv2.COLOR_BGR2RGB
    )
)

plt.title("Overlay")

plt.axis("off")

plt.tight_layout()

plt.show()