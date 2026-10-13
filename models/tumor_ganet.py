"""
Proposed TumorGANet Model Architecture.
Based on Figure 3, Section IV, and Table 3 of the IEEE Access 2024 paper:
"TumorGANet: A Transfer Learning and Generative Adversarial Network-Based Data Augmentation Model for Brain Tumor Classification"

Architecture:
- Base Feature Extractor: Pre-trained VGG-16 (16 layers, 3x3 conv blocks)
- Custom Classification Head:
    * AdaptiveAvgPool2d((7, 7)) / Flatten
    * Dense(1024) + ReLU
    * Dropout(p=0.5)
    * Dense(1024) + ReLU
    * Dense(4) + Softmax (Categorical Cross-Entropy)
"""

from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from data.preprocessor import BrainMRIPreprocessor


class TumorGANet(nn.Module):
    """
    TumorGANet Classification Network (VGG-16 Backbone + Custom 1024 Dense Head).
    """

    def __init__(
        self,
        num_classes: int = config.NUM_CLASSES,
        dropout_rate: float = config.MODEL_CONFIG["dropout_rate"],
        dense_units: int = config.MODEL_CONFIG["dense_units"],
        pretrained: bool = True,
        fine_tune_layers: int = 4
    ):
        super(TumorGANet, self).__init__()
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.dense_units = dense_units

        # 1. Base Pre-trained VGG-16 Convolutional Backbone
        if pretrained:
            weights = models.VGG16_Weights.DEFAULT
            vgg16 = models.vgg16(weights=weights)
        else:
            vgg16 = models.vgg16(weights=None)

        self.features = vgg16.features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        # Freeze earlier feature extraction layers if fine-tuning top blocks
        if pretrained and fine_tune_layers >= 0:
            total_params = len(list(self.features.parameters()))
            freeze_until = max(0, total_params - fine_tune_layers)
            for idx, param in enumerate(self.features.parameters()):
                if idx < freeze_until:
                    param.requires_grad = False
                else:
                    param.requires_grad = True

        # 2. Custom Classification Head matching Figure 3 & Table 3
        # VGG-16 pooled output is 512 channels * 7 * 7 = 25088
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, dense_units),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(dense_units, dense_units),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(dense_units, num_classes)
            # CrossEntropyLoss applies Softmax internally during training;
            # Softmax is exposed via predict_proba() for inference
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass. Returns raw logits (B, 4).
        """
        x = self.features(x)
        x = self.avgpool(x)
        logits = self.classifier(x)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns normalized probability distribution using Softmax.
        """
        logits = self.forward(x)
        return torch.softmax(logits, dim=-1)

    def predict_image(
        self,
        image: Image.Image,
        device: Optional[torch.device] = None
    ) -> Dict[str, Any]:
        """
        High-level inference helper for a single PIL image.
        Returns predicted class name, index, and confidence breakdown.
        """
        if device is None:
            device = next(self.parameters()).device

        self.eval()
        preprocessor = BrainMRIPreprocessor(is_training=False)
        transform = preprocessor.get_torch_transform()
        tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            probs = self.predict_proba(tensor).cpu().numpy().squeeze()

        pred_idx = int(np.argmax(probs))
        pred_class = config.CLASS_LABELS[pred_idx]
        confidence = float(probs[pred_idx])

        breakdown = {
            config.CLASS_LABELS[i]: float(probs[i])
            for i in range(self.num_classes)
        }

        return {
            "predicted_class": pred_class,
            "predicted_index": pred_idx,
            "confidence": confidence,
            "probabilities": breakdown
        }


def build_tumorganet(pretrained: bool = True, device: Optional[str] = None) -> TumorGANet:
    """
    Factory helper to initialize TumorGANet on the target computing device.
    """
    device_obj = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = TumorGANet(
        num_classes=config.NUM_CLASSES,
        dropout_rate=config.MODEL_CONFIG["dropout_rate"],
        dense_units=config.MODEL_CONFIG["dense_units"],
        pretrained=pretrained
    ).to(device_obj)
    return model

