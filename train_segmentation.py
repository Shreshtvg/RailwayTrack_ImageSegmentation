import os
import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp

from torch.utils.data import Dataset, DataLoader

# =========================
# CONFIG
# =========================

TRAIN_IMG_DIR = "dataset/train/images"
TRAIN_MASK_DIR = "dataset/train/masks"

VAL_IMG_DIR = "dataset/valid/images"
VAL_MASK_DIR = "dataset/valid/masks"

BATCH_SIZE = 4
LR = 1e-4
EPOCHS = 50

MODEL_PATH = "outputs/checkpoints/railway_unet_resnet34_best.pth"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

os.makedirs(
    "outputs/checkpoints",
    exist_ok=True
)

print(f"Using Device: {DEVICE}")

# =========================
# DATASET
# =========================


class RailwayDataset(Dataset):

    def __init__(self, image_dir, mask_dir):

        self.image_dir = image_dir
        self.mask_dir = mask_dir

        self.images = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith(
                (".jpg", ".jpeg", ".png")
            )
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image_name = self.images[idx]

        image_path = os.path.join(
            self.image_dir,
            image_name
        )

        mask_name = (
            os.path.splitext(image_name)[0]
            + ".png"
        )

        mask_path = os.path.join(
            self.mask_dir,
            mask_name
        )

        image = cv2.imread(image_path)
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        mask = cv2.imread(
            mask_path,
            cv2.IMREAD_GRAYSCALE
        )

        image = cv2.resize(
            image,
            (512, 512)
        )

        mask = cv2.resize(
            mask,
            (512, 512)
        )

        image = image.astype(
            np.float32
        ) / 255.0

        mask = (
            mask > 127
        ).astype(np.float32)

        image = torch.tensor(
            image.transpose(2, 0, 1),
            dtype=torch.float32
        )

        mask = torch.tensor(
            mask,
            dtype=torch.float32
        ).unsqueeze(0)

        return image, mask


# =========================
# METRICS
# =========================

def dice_score(preds, masks):

    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()

    intersection = (
        preds * masks
    ).sum()

    return (
        2.0 * intersection + 1e-8
    ) / (
        preds.sum()
        + masks.sum()
        + 1e-8
    )


def iou_score(preds, masks):

    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()

    intersection = (
        preds * masks
    ).sum()

    union = (
        preds.sum()
        + masks.sum()
        - intersection
    )

    return (
        intersection + 1e-8
    ) / (
        union + 1e-8
    )


# =========================
# TRAIN
# =========================

def main():

    train_dataset = RailwayDataset(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR
    )

    val_dataset = RailwayDataset(
        VAL_IMG_DIR,
        VAL_MASK_DIR
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1
    )

    model.to(DEVICE)

    bce_loss = torch.nn.BCEWithLogitsLoss()

    dice_loss = smp.losses.DiceLoss(
        mode="binary"
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR
    )

    scaler = torch.amp.GradScaler("cuda")

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):

        # ------------------
        # TRAIN
        # ------------------

        model.train()

        train_loss = 0

        for images, masks in train_loader:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            masks = masks.to(
                DEVICE,
                non_blocking=True
            )

            optimizer.zero_grad()

            with torch.amp.autocast("cuda"):

                outputs = model(images)

                loss = (
                    bce_loss(outputs, masks)
                    +
                    dice_loss(outputs, masks)
                )

            scaler.scale(loss).backward()

            scaler.step(optimizer)

            scaler.update()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ------------------
        # VALIDATION
        # ------------------

        model.eval()

        val_loss = 0
        dice_total = 0
        iou_total = 0

        with torch.no_grad():

            for images, masks in val_loader:

                images = images.to(DEVICE)
                masks = masks.to(DEVICE)

                outputs = model(images)

                loss = (
                    bce_loss(outputs, masks)
                    +
                    dice_loss(outputs, masks)
                )

                val_loss += loss.item()

                dice_total += dice_score(
                    outputs,
                    masks
                ).item()

                iou_total += iou_score(
                    outputs,
                    masks
                ).item()

        val_loss /= len(val_loader)

        mean_dice = (
            dice_total /
            len(val_loader)
        )

        mean_iou = (
            iou_total /
            len(val_loader)
        )

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Dice: {mean_dice:.4f} | "
            f"IoU: {mean_iou:.4f}"
        )

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                model.state_dict(),
                MODEL_PATH
            )

            print(
                "✓ Best Model Saved"
            )

    print("\nTraining Finished")


if __name__ == "__main__":
    main()