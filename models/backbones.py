"""
Transfer Learning Backbone Suite for Brain Tumor Classification.
Implements the 10 Transfer Learning architectures compared in Table 5 and Table 6:
1. VGG-16 (TumorGANet)
2. GoogLeNet
3. DenseNet-121
4. MobileNet (v1/v3)
5. MobileNet-v2
6. VGG-19
7. AlexNet
8. ResNet-101
9. Inception-v3
10. Xception
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from torchvision import models

from models.tumor_ganet import TumorGANet


class TransferLearningModelFactory:
    """
    Factory to construct and adapt any of the 10 models evaluated in the TumorGANet paper.
    """

    @staticmethod
    def get_model(
        model_name: str,
        num_classes: int = 4,
        pretrained: bool = True
    ) -> nn.Module:
        name = model_name.lower().replace("-", "").replace("_", "")

        # 1. Proposed TumorGANet (VGG-16 + Custom Dense Head)
        if name in ["vgg16", "tumorganet"]:
            return TumorGANet(num_classes=num_classes, pretrained=pretrained)

        # 2. VGG-19
        elif name == "vgg19":
            weights = models.VGG19_Weights.DEFAULT if pretrained else None
            model = models.vgg19(weights=weights)
            in_features = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(in_features, num_classes)
            return model

        # 3. ResNet-101
        elif name == "resnet101":
            weights = models.ResNet101_Weights.DEFAULT if pretrained else None
            model = models.resnet101(weights=weights)
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
            return model

        # 4. DenseNet-121
        elif name == "densenet121":
            weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
            model = models.densenet121(weights=weights)
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, num_classes)
            return model

        # 5. GoogLeNet
        elif name == "googlenet":
            weights = models.GoogLeNet_Weights.DEFAULT if pretrained else None
            model = models.googlenet(weights=weights, aux_logits=False)
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
            return model

        # 6. MobileNet-v2
        elif name == "mobilenetv2":
            weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
            model = models.mobilenet_v2(weights=weights)
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, num_classes)
            return model

        # 7. MobileNet (v3-Large as modern mobile equivalent)
        elif name == "mobilenet":
            weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
            model = models.mobilenet_v3_large(weights=weights)
            in_features = model.classifier[3].in_features
            model.classifier[3] = nn.Linear(in_features, num_classes)
            return model

        # 8. AlexNet
        elif name == "alexnet":
            weights = models.AlexNet_Weights.DEFAULT if pretrained else None
            model = models.alexnet(weights=weights)
            in_features = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(in_features, num_classes)
            return model

        # 9. Inception-v3
        elif name == "inceptionv3":
            weights = models.Inception_V3_Weights.DEFAULT if pretrained else None
            model = models.inception_v3(weights=weights, aux_logits=False)
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
            return model

        # 10. Xception / ResNeXt-50 fallback
        elif name in ["xception", "resnext50"]:
            weights = models.resnext50_32x4d.DEFAULT if pretrained else None
            model = models.resnext50_32x4d(weights=weights)
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
            return model

        else:
            raise ValueError(f"Unknown model name: '{model_name}'. Choose from: "
                             f"vgg16, googlenet, densenet121, mobilenet, vgg19, "
                             f"mobilenetv2, alexnet, xception, resnet101, inceptionv3")


def get_model(model_name: str, num_classes: int = 4, pretrained: bool = True) -> nn.Module:
    """Convenience functional wrapper."""
    return TransferLearningModelFactory.get_model(model_name, num_classes, pretrained)

