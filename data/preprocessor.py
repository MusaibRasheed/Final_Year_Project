"""
Data Preprocessing Module for TumorGANet Brain Tumor Classification.
Implements the exact mathematical formulas from Section V-A of the paper:
- Image Resizing (512x512 to 224x224): Equations (2)-(3)
- Image Scaling / Normalization: Equations (4)-(5)
- Image Zooming (20% range): Equation (6)
- Horizontal Flipping: Equation (7)
- Shearing (shear_range = 0.2)
"""

import math
import random
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
from torchvision.transforms import functional as TF
from typing import Tuple, Union, Optional

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


class BrainMRIPreprocessor:
    """
    Mathematical and Torch-based implementation of the TumorGANet preprocessing pipeline.
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = config.IMAGE_SIZE,
        scale_a: float = config.NORM_SCALE_A,
        scale_b: float = config.NORM_SCALE_B,
        shear_range: float = config.SHEAR_RANGE,
        zoom_range: float = config.ZOOM_RANGE,
        horizontal_flip: bool = config.HORIZONTAL_FLIP,
        is_training: bool = True
    ):
        self.target_size = target_size
        self.scale_a = scale_a
        self.scale_b = scale_b
        self.shear_range = shear_range
        self.zoom_range = zoom_range
        self.horizontal_flip = horizontal_flip
        self.is_training = is_training

    @staticmethod
    def resize_image(image: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """
        Equations (2) & (3):
        SF_w = W_new / W_original, SF_h = H_new / H_original
        x_new = x * SF_w, y_new = y * SF_h
        """
        pil_img = Image.fromarray(image.astype(np.uint8)) if isinstance(image, np.ndarray) else image
        resized_pil = pil_img.resize(target_size, Image.BILINEAR)
        return np.array(resized_pil)

    @staticmethod
    def normalize_pixels(image: np.ndarray, a: float = 1.0, b: float = 0.0) -> np.ndarray:
        """
        Equation (4): NormalizedPixelValue = OriginalPixelValue / 255
        Equation (5): TransformedPixelValue = a * OriginalPixelValue + b
        Combined to map [0, 255] to [0.0, 1.0] with contrast and brightness control.
        """
        normalized = (image.astype(np.float32) / 255.0)
        transformed = a * normalized + b
        return np.clip(transformed, 0.0, 1.0)

    @staticmethod
    def random_zoom(image: np.ndarray, zoom_range: float = 0.2) -> np.ndarray:
        """
        Equation (6):
        ZoomedPixelValue = OriginalPixelValue * (1 + ZoomRange * (RandomValue - 0.5))
        """
        random_val = random.random()
        zoom_factor = 1.0 + zoom_range * (random_val - 0.5)

        h, w = image.shape[:2]
        new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)

        pil_img = Image.fromarray(image.astype(np.uint8)) if image.dtype != np.uint8 else Image.fromarray(image)
        zoomed = pil_img.resize((new_w, new_h), Image.BILINEAR)

        # Crop or pad back to original dimensions (h, w)
        if zoom_factor > 1.0:
            left = (new_w - w) // 2
            top = (new_h - h) // 2
            zoomed = zoomed.crop((left, top, left + w, top + h))
        else:
            padded = Image.new(zoomed.mode, (w, h), color=0)
            left = (w - new_w) // 2
            top = (h - new_h) // 2
            padded.paste(zoomed, (left, top))
            zoomed = padded

        return np.array(zoomed)

    @staticmethod
    def horizontal_flip_image(image: np.ndarray) -> np.ndarray:
        """
        Equation (7):
        FlippedPixelValue(i, j) = OriginalPixelValue(i, Width - 1 - j)
        """
        return np.fliplr(image)

    @staticmethod
    def random_shear(image: np.ndarray, shear_range: float = 0.2) -> np.ndarray:
        """
        Applies shear transformation along horizontal axis.
        """
        shear_angle = random.uniform(-shear_range * 45, shear_range * 45)
        pil_img = Image.fromarray(image.astype(np.uint8)) if isinstance(image, np.ndarray) else image
        sheared = TF.affine(
            pil_img,
            angle=0,
            translate=[0, 0],
            scale=1.0,
            shear=[shear_angle, 0],
            interpolation=T.InterpolationMode.BILINEAR
        )
        return np.array(sheared)

    def preprocess_numpy(self, img_array: np.ndarray) -> np.ndarray:
        """
        Full step-by-step pipeline for an input image (H, W, C) -> (224, 224, C).
        Matches Figure 4 in the paper.
        """
        # 1. Resize to target (224, 224)
        processed = self.resize_image(img_array, self.target_size)

        if self.is_training:
            # 2. Random Shear
            if random.random() > 0.5:
                processed = self.random_shear(processed, self.shear_range)

            # 3. Random Zoom
            if random.random() > 0.5:
                processed = self.random_zoom(processed, self.zoom_range)

            # 4. Random Horizontal Flip
            if self.horizontal_flip and random.random() > 0.5:
                processed = self.horizontal_flip_image(processed)

        # 5. Normalization / Pixel Scaling
        normalized = self.normalize_pixels(processed, self.scale_a, self.scale_b)
        return normalized

    def get_torch_transform(self) -> T.Compose:
        """
        Returns PyTorch transforms equivalent to the paper's specification.
        """
        if self.is_training:
            return T.Compose([
                T.Resize(self.target_size),
                T.RandomHorizontalFlip(p=0.5 if self.horizontal_flip else 0.0),
                T.RandomAffine(
                    degrees=0,
                    shear=(-self.shear_range * 45, self.shear_range * 45),
                    scale=(1.0 - self.zoom_range / 2, 1.0 + self.zoom_range / 2)
                ),
                T.ToTensor(),  # Automatically scales pixels to [0, 1]
                # ImageNet standard normalization for pre-trained backbones
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return T.Compose([
                T.Resize(self.target_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])


# Demonstration test helper
if __name__ == "__main__":
    dummy = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    preprocessor = BrainMRIPreprocessor(is_training=True)
    out = preprocessor.preprocess_numpy(dummy)
    print("Preprocessing successful! Output shape:", out.shape, "Range:", out.min(), "to", out.max())

