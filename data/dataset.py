"""
Dataset loader and generator for TumorGANet Brain Tumor Classification.
Supports:
1. Loading the Kaggle Brain Tumor MRI Dataset (7023 images, 4 classes).
2. Automatic generation of realistic synthetic / sample MRI scans for quick prototyping & testing.
3. Train/Validation/Test split loaders with PyTorch DataLoader.
"""

import os
import glob
from pathlib import Path
from typing import Tuple, Optional, List, Dict
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from data.preprocessor import BrainMRIPreprocessor


class BrainMRIDataset(Dataset):
    """
    PyTorch Dataset for Brain Tumor MRI classification.
    Expects directory structure:
        <root_dir>/
            glioma/
            meningioma/
            notumor/
            pituitary/
    """

    def __init__(
        self,
        root_dir: Path,
        transform: Optional[T.Compose] = None,
        classes: List[str] = config.CLASSES
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.classes = classes
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.image_paths = []
        self.labels = []

        self._load_dataset()

    def _load_dataset(self):
        for cls_name in self.classes:
            cls_dir = self.root_dir / cls_name
            if not cls_dir.exists():
                continue
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.tif"]:
                for file_path in cls_dir.glob(ext):
                    self.image_paths.append(file_path)
                    self.labels.append(self.class_to_idx[cls_name])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        else:
            default_tf = T.Compose([
                T.Resize(config.IMAGE_SIZE),
                T.ToTensor()
            ])
            image = default_tf(image)

        return image, label


def create_synthetic_mri_slice(
    tumor_type: str,
    size: Tuple[int, int] = (224, 224),
    noise_level: float = 0.05
) -> Image.Image:
    """
    Generates a realistic synthetic axial T1/T2-weighted brain MRI slice.
    Includes:
    - Skull / Calvarium oval ring
    - Brain parenchyma with bilateral hemispheres and central sulcus
    - Ventricles (butterfly shape)
    - Specific tumor anomalies:
        * Glioma: Infiltrative, heterogeneous lesion in the cerebral hemisphere
        * Meningioma: Well-demarcated dural-based extra-axial mass with homogeneous enhancement
        * Pituitary: Sellar / suprasellar mass near the skull base / optic chiasm
        * No Tumor: Symmetric, normal anatomy
    """
    w, h = size
    img = Image.new("L", (w, h), color=15)
    draw = ImageDraw.Draw(img)

    cx, cy = w // 2, h // 2
    rx, ry = int(w * 0.40), int(h * 0.44)

    # 1. Outer skull & subcutaneous scalp
    draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=45, outline=180, width=4)

    # 2. Brain Parenchyma (cortex & white matter)
    inner_rx, inner_ry = int(rx * 0.92), int(ry * 0.92)
    draw.ellipse([cx - inner_rx, cy - inner_ry, cx + inner_rx, cy + inner_ry], fill=90)

    # 3. Interhemispheric fissure (falx cerebri line)
    draw.line([cx, cy - inner_ry + 10, cx, cy + inner_ry - 10], fill=40, width=2)

    # 4. Lateral Ventricles (Cerebrospinal Fluid - Dark in T1)
    v_w, v_h = int(w * 0.08), int(h * 0.16)
    # Left frontal horn
    draw.ellipse([cx - v_w - 6, cy - v_h, cx - 6, cy + v_h], fill=30)
    # Right frontal horn
    draw.ellipse([cx + 6, cy - v_h, cx + v_w + 6, cy + v_h], fill=30)

    # 5. Tumor Pathology Specifics
    if tumor_type == "glioma":
        # Infiltrative, irregular hyperintensity in frontal/temporal white matter
        t_cx, t_cy = cx - int(w * 0.18), cy - int(h * 0.12)
        t_r = int(w * 0.14)
        draw.ellipse([t_cx - t_r, t_cy - t_r, t_cx + t_r, t_cy + t_r], fill=210)
        # Irregular necrotic center
        draw.ellipse([t_cx - int(t_r * 0.4), t_cy - int(t_r * 0.4),
                      t_cx + int(t_r * 0.4), t_cy + int(t_r * 0.4)], fill=120)

    elif tumor_type == "meningioma":
        # Dural based, well-circumscribed, peripheral bright mass
        t_cx, t_cy = cx + int(inner_rx * 0.70), cy + int(h * 0.05)
        t_r = int(w * 0.12)
        draw.ellipse([t_cx - t_r, t_cy - t_r, t_cx + t_r, t_cy + t_r], fill=240, outline=255, width=2)

    elif tumor_type == "pituitary":
        # Sellar mass, midline near center / skull base
        t_cx, t_cy = cx, cy + int(h * 0.22)
        t_r = int(w * 0.09)
        draw.ellipse([t_cx - t_r, t_cy - t_r, t_cx + t_r, t_cy + t_r], fill=235)

    # Blur slightly for organic medical scan appearance
    img = img.filter(ImageFilter.GaussianBlur(radius=1.2))

    # Add Rician/Gaussian noise typical of MRI acquisition
    arr = np.array(img, dtype=np.float32)
    noise = np.random.normal(0, noise_level * 255, arr.shape)
    noisy_arr = np.clip(arr + noise, 0, 255).astype(np.uint8)

    # Convert to 3-channel RGB as expected by CNN backbones
    rgb_img = Image.fromarray(noisy_arr).convert("RGB")
    return rgb_img


def generate_sample_dataset(output_dir: Path, samples_per_class: int = 25):
    """
    Creates a sample / mock dataset with standard structure for immediate training and verification.
    """
    output_dir = Path(output_dir)
    train_dir = output_dir / "Training"
    test_dir = output_dir / "Testing"

    for split_dir, count in [(train_dir, samples_per_class), (test_dir, max(5, samples_per_class // 4))]:
        for cls_name in config.CLASSES:
            cls_folder = split_dir / cls_name
            cls_folder.mkdir(parents=True, exist_ok=True)
            for i in range(count):
                mri_img = create_synthetic_mri_slice(cls_name)
                mri_img.save(cls_folder / f"sample_{cls_name}_{i:04d}.jpg")

    print(f"Sample Brain MRI dataset created at: {output_dir}")


def get_data_loaders(
    data_dir: Path,
    batch_size: int = config.MODEL_CONFIG["batch_size"],
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader]:
    """
    Builds training and testing DataLoaders.
    If data_dir doesn't contain samples yet, generates a starter sample set automatically.
    """
    data_dir = Path(data_dir)
    train_dir = data_dir / "Training"
    test_dir = data_dir / "Testing"

    # Fallback to sample generation if directory is empty
    if not train_dir.exists() or len(list(train_dir.glob("*/*.*"))) == 0:
        print(f"No existing dataset found at {data_dir}. Generating verified sample brain MRI dataset...")
        generate_sample_dataset(data_dir, samples_per_class=30)

    preprocessor = BrainMRIPreprocessor()
    train_transform = preprocessor.get_torch_transform()
    test_preprocessor = BrainMRIPreprocessor(is_training=False)
    test_transform = test_preprocessor.get_torch_transform()

    train_dataset = BrainMRIDataset(train_dir, transform=train_transform)
    test_dataset = BrainMRIDataset(test_dir, transform=test_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True if len(train_dataset) > batch_size else False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, test_loader

