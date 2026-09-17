"""
Configuration settings for TumorGANet Brain Tumor Classification System.
Based on the IEEE Access 2024 paper:
"TumorGANet: A Transfer Learning and Generative Adversarial Network-Based Data Augmentation Model for Brain Tumor Classification"
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SYNTHETIC_DATA_DIR = DATA_DIR / "synthetic"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
RESULTS_DIR = BASE_DIR / "results"
EXPLANATIONS_DIR = RESULTS_DIR / "explanations"

# Create required directories
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, SYNTHETIC_DATA_DIR,
                  CHECKPOINTS_DIR, RESULTS_DIR, EXPLANATIONS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Dataset Specs (Table 2 in paper)
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
CLASS_LABELS = {
    0: "glioma",
    1: "meningioma",
    2: "notumor",
    3: "pituitary"
}
LABEL_TO_IDX = {v: k for k, v in CLASS_LABELS.items()}
NUM_CLASSES = len(CLASSES)

DATASET_DISTRIBUTION = {
    "training": {
        "glioma": 1321,
        "meningioma": 1339,
        "notumor": 1595,
        "pituitary": 1457,
        "total": 5712
    },
    "testing": {
        "glioma": 300,
        "meningioma": 306,
        "notumor": 405,
        "pituitary": 300,
        "total": 1311
    },
    "total_images": 7023
}

# Image Preprocessing Hyperparameters (Equations 2-7)
ORIGINAL_IMAGE_SIZE = (512, 512)
IMAGE_SIZE = (224, 224)
IMAGE_CHANNELS = 3
NORM_SCALE_A = 1.00   # Eq. 5: Brightness control parameter
NORM_SCALE_B = 0.00   # Eq. 5: Contrast control parameter
SHEAR_RANGE = 0.2     # Section V-A
ZOOM_RANGE = 0.2      # Eq. 6: 20% random zooming
HORIZONTAL_FLIP = True # Eq. 7

# GAN Hyperparameters (Section V-B & Algorithm 1)
GAN_CONFIG = {
    "latent_dim": 100,
    "image_size": 224,
    "channels": 3,
    "batch_size": 32,
    "epochs": 10000,   # Training history in Fig. 6 plotted up to 10k iterations
    "lr_g": 0.0002,
    "lr_d": 0.0002,
    "beta1": 0.5,
    "beta2": 0.999
}

# TumorGANet (VGG-16 Backbone) Hyperparameters (Table 3 in paper)
MODEL_CONFIG = {
    "backbone": "vgg16",
    "optimizer": "Adam",
    "learning_rate": 0.0001,     # 1e-4
    "dropout_rate": 0.5,         # 0.5
    "dense_units": 1024,         # Two FC layers of 1024 as in Fig. 3
    "activation": "relu",
    "output_activation": "softmax",
    "batch_size": 32,
    "epochs": 50
}

# 10 Compared Transfer Learning Models (Table 5)
COMPARISON_MODELS = [
    "vgg16",           # Proposed TumorGANet
    "googlenet",
    "densenet121",
    "mobilenet",
    "vgg19",
    "mobilenet_v2",
    "alexnet",
    "xception",
    "resnet101",
    "inception_v3"
]

# Ablation Grid (Table 7)
ABLATION_OPTIMIZERS = ["Adam", "SGD", "Adagrad", "Nadam", "RMSprop"]
ABLATION_LEARNING_RATES = [0.001, 0.0001, 0.00001]
ABLATION_EPOCHS = 50

# Paper Reported Benchmarks for TumorGANet (Table 5 & 8)
PAPER_BENCHMARKS = {
    "TumorGANet": {
        "accuracy": 99.53,
        "precision": 100.0,
        "recall": 100.0,
        "f1_score": 99.0,
        "hamming_loss": 0.20,
        "mcc": 99.50,
        "jaccard_score": 98.50,
        "cohens_kappa": 99.50
    }
}

