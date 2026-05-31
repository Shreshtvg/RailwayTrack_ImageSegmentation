import os
import cv2
import torch
import numpy as np

from torch.utils.data import Dataset


class RailwayDataset(Dataset):

    def __init__(self, image_dir, mask_dir):

        self.image_dir = image_dir
        self.mask_dir = mask_dir

        self.images = sorted([
            f for f in os.listdir(image_dir)
            if f.endswith((".jpg", ".png", ".jpeg"))
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img_name = self.images[idx]

        image_path = os.path.join(
            self.image_dir,
            img_name
        )

        mask_name = os.path.splitext(img_name)[0] + ".png"

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

        image = cv2.resize(image, (512, 512))
        mask = cv2.resize(mask, (512, 512))

        image = image.astype(np.float32) / 255.0

        mask = (mask > 127).astype(np.float32)

        image = torch.tensor(
            image.transpose(2, 0, 1),
            dtype=torch.float32
        )

        mask = torch.tensor(
            mask,
            dtype=torch.float32
        ).unsqueeze(0)

        return image, mask