"""
ResNet-50 Feature Extractor for Brain MRI Visual Representation Analysis.
Implements the feature extraction module described in Section V-C1 of the TumorGANet paper.
Captures 2048-dimensional hierarchical visual embeddings from the penultimate layer of ResNet-50.
"""

from pathlib import Path
from typing import Tuple, List, Optional
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from data.preprocessor import BrainMRIPreprocessor


class ResNet50FeatureExtractor(nn.Module):
    """
    Truncated ResNet-50 feature extractor.
    Removes the final classification FC layer, outputting 2048-dimensional embeddings.
    """

    def __init__(self, pretrained: bool = True, device: Optional[str] = None):
        super(ResNet50FeatureExtractor, self).__init__()
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        # Load ResNet-50
        if pretrained:
            weights = models.ResNet50_Weights.DEFAULT
            base_model = models.resnet50(weights=weights)
        else:
            base_model = models.resnet50(weights=None)

        # Retain all layers up to Global Average Pooling (conv1 through avgpool)
        self.features = nn.Sequential(*list(base_model.children())[:-1])
        self.embedding_dim = 2048

        # Freeze feature extraction backbone
        for param in self.features.parameters():
            param.requires_grad = False

        self.to(self.device)
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: (B, 3, 224, 224)
        Output: (B, 2048) feature embedding vector
        """
        x = x.to(self.device)
        with torch.no_grad():
            feat = self.features(x)
            feat = torch.flatten(feat, 1)  # (B, 2048)
        return feat

    def extract_image_embedding(self, image: Image.Image) -> np.ndarray:
        """
        Extracts embedding vector for a single PIL image.
        Returns numpy array of shape (2048,).
        """
        preprocessor = BrainMRIPreprocessor(is_training=False)
        transform = preprocessor.get_torch_transform()
        tensor = transform(image).unsqueeze(0).to(self.device)

        embedding = self.forward(tensor)
        return embedding.cpu().numpy().squeeze()

    def extract_batch(self, images: List[Image.Image]) -> np.ndarray:
        """
        Extracts embeddings for a batch of PIL images.
        Returns numpy array of shape (N, 2048).
        """
        preprocessor = BrainMRIPreprocessor(is_training=False)
        transform = preprocessor.get_torch_transform()
        batch_tensors = torch.stack([transform(img) for img in images]).to(self.device)

        embeddings = self.forward(batch_tensors)
        return embeddings.cpu().numpy()

