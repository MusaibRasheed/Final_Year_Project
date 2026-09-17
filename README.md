# TumorGANet: Brain Tumor Classification & Explainable AI

[![Paper: IEEE Access 2024](https://img.shields.io/badge/IEEE%20Access-10.1109%2FACCESS.2024.3429633-blue)](https://doi.org/10.1109/ACCESS.2024.3429633)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-green.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)

> **Official Implementation from Scratch** of the research paper:  
> **"TumorGANet: A Transfer Learning and Generative Adversarial Network-Based Data Augmentation Model for Brain Tumor Classification"**  
> *Anindya Nag, Hirak Mondal, Md Mehedi Hassan, Taher Al-Shehari, Mohammed Kadrie, Muna Al-Razgan, Taha Alfakih, Sujit Biswas, Anupam Kumar Bairagi*  
> **IEEE Access**, Vol. 12, pp. 103060–103081, 2024. DOI: `10.1109/ACCESS.2024.3429633`

---

## 📑 Table of Contents
1. [Overview & Clinical Significance](#-overview--clinical-significance)
2. [TumorGANet Architecture](#-tumorganet-architecture)
3. [Mathematical Formulations](#-mathematical-formulations)
4. [Dataset Specifications](#-dataset-specifications)
5. [Directory Structure](#-directory-structure)
6. [Quick Start & Installation](#-quick-start--installation)
7. [Running the Project](#-running-the-project)
8. [Clinical Web UI & Standalone Viewer](#-clinical-web-ui--standalone-viewer)
9. [Published Benchmark Results](#-published-benchmark-results)
10. [Explainable AI (LIME)](#-explainable-ai-lime)
11. [Citation](#-citation)

---

## 🔬 Overview & Clinical Significance

Accurate brain tumor diagnosis via Magnetic Resonance Imaging (MRI) is critical for oncology treatment planning and neurosurgical guidance. However, clinical automated systems face significant challenges:
- **High morphological variability**: Tumors differ drastically in location, intensity, and shape.
- **Data imbalance & scarcity**: Pathological positive scans are often limited compared to non-tumorous cases.
- **Black-box opacity**: Deep neural networks lack transparent justification for clinical decision-making.

**TumorGANet** addresses these limitations by introducing:
1. **GAN-based Data Augmentation**: A Deep Convolutional GAN (DCGAN) synthesizes diverse, high-fidelity brain MRI scans to balance training distributions and prevent overfitting.
2. **Hierarchical Feature Extraction**: Utilizes **ResNet-50** to extract dense 2048-dimensional visual embeddings.
3. **Transfer Learning Classifier**: Integrates pre-trained **VGG-16** with a customized dense classifier (`Dense(1024) -> Dropout(0.5) -> Dense(1024) -> Softmax(4)`), achieving **99.53% accuracy**, **100% precision/recall**, and a **0.2% Hamming Loss**.
4. **Explainable AI (XAI)**: Local Interpretable Model-Agnostic Explanations (**LIME**) isolates super-pixel tumor regions, traces yellow boundary contours, and renders probability heatmaps for clinician validation.

---

## 🏛️ TumorGANet Architecture

```
Brain MRI Scans (512x512)
        │
        ▼
[1. Mathematical Preprocessing]
  ├── Bilinear Resizing: (512x512) ➔ (224x224x3) [Eq. 2-3]
  ├── Pixel Normalization: I / 255.0 [Eq. 4-5]
  └── Augmentations: Shearing (0.2), Zooming (0.2) [Eq. 6], Flip [Eq. 7]
        │
        ├────────────────────────────────────────┐
        ▼                                        ▼
[2. ResNet-50 Feature Extractor]      [3. DCGAN Data Augmentation]
  Penultimate Hierarchical Layer        Generator (G) vs Discriminator (D)
  2048-dimensional embeddings           Minimax loss [Eq. 8]
        │                               Synthesizes balanced MRI slices
        └───────────────────┬────────────────────┘
                            ▼
           [Augmented Training Dataset (Real + Synthetic)]
                            │
                            ▼
              [4. Proposed TumorGANet Backbone]
                 Pre-trained VGG-16 Base (16 Conv layers)
                            │
                            ▼
                 AdaptiveAvgPool2d((7, 7)) / Flatten
                            │
                            ▼
                 Dense Layer (1024 Units, ReLU)
                            │
                            ▼
                 Dropout Regularization (p = 0.5) [Eq. 11]
                            │
                            ▼
                 Dense Layer (1024 Units, ReLU)
                            │
                            ▼
                 Dropout Regularization (p = 0.5)
                            │
                            ▼
                 Softmax Output Layer (4 Classes: Glioma, Meningioma, Pituitary, No Tumor)
                            │
        ┌───────────────────┴───────────────────┐
        ▼                                       ▼
[5. Clinical Performance Metrics]       [6. Explainable AI (LIME)]
  - Accuracy: 99.53% [Eq. 13]             - Super-pixel Segmentation
  - Hamming Loss: 0.20% [Eq. 17]          - Correct Region Mask
  - Matthews Corr. (MCC): 99.50% [Eq. 19] - Yellow Highlight Contour
  - Jaccard Score: 98.50% [Eq. 18]        - Activation Heatmaps [Eq. 21]
  - Cohen's Kappa: 99.50% [Eq. 20]
```

---

## 📐 Mathematical Formulations

### 1. Image Preprocessing & Geometric Augmentation
- **Resizing Coordinate Mapping (Equations 2 & 3)**:
  $$SF_w = \frac{W_{new}}{W_{original}}, \quad SF_h = \frac{H_{new}}{H_{original}}$$
  $$x_{new} = x \times SF_w, \quad y_{new} = y \times SF_h$$
- **Pixel Normalization & Scaling (Equations 4 & 5)**:
  $$\text{NormalizedPixelValue} = \frac{\text{OriginalPixelValue}}{255}$$
  $$\text{TransformedPixelValue} = a \times \text{OriginalPixelValue} + b \quad (a = 1.00, \; b = 0)$$
- **Image Zooming (Equation 6)**:
  $$\text{ZoomedPixelValue} = \text{OriginalPixelValue} \times \left(1 + \text{ZoomRange} \times (\text{RandomValue} - 0.5)\right)$$
- **Horizontal Flipping (Equation 7)**:
  $$\text{FlippedPixelValue}(i, j) = \text{OriginalPixelValue}(i, \; \text{Width} - 1 - j)$$

### 2. Generative Adversarial Network (DCGAN)
- **Minimax Objective Function (Equation 8)**:
  $$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{data}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))]$$

### 3. Optimizer Update Rules
- **Adam Parameter Vector Update (Equation 10)**:
  $$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \cdot \hat{m}_t$$
- **Dropout Masking (Equation 11)**:
  $$\text{Output} = \text{Input} \odot \text{mask}, \quad \text{mask} \sim \text{Bernoulli}(1 - p)$$
- **ReLU Activation (Equation 12)**:
  $$\text{ReLU}(x) = \max(0, x)$$

### 4. Clinical Evaluation Metrics
- **Accuracy (Equation 13)**: $\frac{TP + TN}{TP + FP + TN + FN}$
- **Precision (Equation 14)**: $\frac{TP}{TP + FP}$
- **Recall (Equation 15)**: $\frac{TP}{TP + FN}$
- **F1-Score (Equation 16)**: $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$
- **Hamming Loss (Equation 17)**: $HL = \frac{1}{n} \sum_{i=1}^n \mathbb{I}[y_i \neq \hat{y}_i]$
- **Jaccard Score (Equation 18)**: $JS = \frac{|X \cap Y|}{|X \cup Y|}$
- **Matthews Correlation Coefficient (Equation 19)**:
  $$MCC = \frac{TP \times TN - FP \times FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$$
- **Cohen's Kappa (Equation 20)**: $CK = \frac{P_o - P_e}{1 - P_e}$

### 5. Local Interpretable Model-Agnostic Explanations (LIME)
- **Objective Function (Equation 21)**:
  $$\mathcal{L}(x, f, \Pi) = \sum_{i=1}^d \Pi_i(x) f_i(x) + C |\Pi|_1$$

---

## 📊 Dataset Specifications

The dataset used is the publicly accessible **Brain Tumor MRI Dataset** from Kaggle:
- **Source**: [Kaggle Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) (`masoudnickparvar/brain-tumor-mri-dataset`)
- **Total Images**: **7,023** MRI scans
- **Distribution (Table 2 in Paper)**:

| Tumor Type | Training Scans | Testing Scans | Total Scans |
| :--- | :---: | :---: | :---: |
| **Glioma** | 1,321 | 300 | 1,621 |
| **Meningioma** | 1,339 | 306 | 1,645 |
| **Pituitary** | 1,457 | 300 | 1,757 |
| **No Tumor** | 1,595 | 405 | 2,000 |
| **Total** | **5,712** | **1,311** | **7,023** |

---

## 📁 Directory Structure

```
Project Build from AntiGravity/
├── config.py                      # Global parameters, hyperparameters & paths
├── requirements.txt               # Dependencies
├── README.md                      # Comprehensive documentation
│
├── data/
│   ├── dataset.py                 # Dataset loader, Kaggle downloader & synthetic generator
│   ├── preprocessor.py            # Preprocessing implementation (Equations 2-7)
│   ├── raw/                       # Real Brain MRI dataset directory
│   └── synthetic/                 # GAN-generated augmented MRI samples
│
├── models/
│   ├── tumor_ganet.py             # Proposed TumorGANet (VGG-16 + 1024 Head)
│   ├── gan.py                     # DCGAN Generator & Discriminator (Equation 8)
│   ├── feature_extractor.py       # ResNet-50 2048-dim feature extraction
│   └── backbones.py               # Transfer learning suite (10 compared models)
│
├── training/
│   ├── train_classifier.py        # Supervised training loop & checkpoints
│   ├── train_gan.py               # GAN training script (Algorithm 1)
│   └── ablation.py                # Full ablation engine (Table 7)
│
├── evaluation/
│   ├── metrics.py                 # Clinical evaluation metrics (Equations 13-20)
│   ├── explainability.py          # LIME XAI pipeline (Figures 11 & 12)
│   └── visualizer.py              # Figures 6, 7, 8, 9, 10 plotting toolkit
│
├── app/
│   ├── app.py                     # Interactive Streamlit Clinical Dashboard
│   └── standalone_viewer.html     # Zero-dependency browser-based Clinical Workstation
│
├── notebooks_and_scripts/
│   ├── run_pipeline.py            # End-to-end CLI pipeline orchestrator
│   └── test_suite.py              # Comprehensive unit & integration tests
│
├── checkpoints/                   # Saved model weights
└── results/                       # Generated plots, confusion matrices & ablation tables
```

---

## 🚀 Quick Start & Installation

### 1. Clone or Open the Workspace
```bash
cd "c:\Users\Musaib\Documents\Project Build from AntiGravity"
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 💻 Running the Project

### 1. Run the Comprehensive Test Suite
Validates all math equations, tensor dimensions, GAN generators, metric calculations, and LIME explanations:
```bash
python notebooks_and_scripts/test_suite.py
```

### 2. Run the End-to-End Pipeline
Executes preprocessing, DCGAN augmentation, TumorGANet training, metric logging, ablation export, and LIME generation:
```bash
python notebooks_and_scripts/run_pipeline.py --epochs 50 --gan_epochs 10
```

### 3. Train the DCGAN Generator
```bash
python training/train_gan.py
```

### 4. Run the Table 7 Optimizer Ablation Study
```bash
python training/ablation.py
```

---

## 🌐 Clinical Web UI & Standalone Viewer

### Option A: Interactive Streamlit Clinical Dashboard
```bash
streamlit run app/app.py
```
**Features**:
- **Diagnosis**: Upload any brain MRI (JPEG/PNG) or pick from clinical sample scans.
- **Explainable AI (LIME)**: Instant rendering of Superpixel Correct Regions, Highlight Contours, and Heatmaps.
- **GAN Studio**: Synthesize novel brain MRI slices using interactive latent noise sampling.
- **Ablation & Benchmarks**: Interactive inspection of Tables 5, 6, and 7.
- **Preprocessing Lab**: Interactive visualization of scaling, shearing, zooming, and horizontal flipping.

### Option B: Zero-Dependency Standalone Browser Viewer
Open `app/standalone_viewer.html` directly in **Google Chrome, Microsoft Edge, or Firefox**:
- No terminal or active Python server required!
- Features an interactive tumor simulator, dynamic LIME toggles, live Chart.js benchmark graphs, and searchable data tables.

---

## 🏆 Published Benchmark Results

### 1. Comparison of 10 Transfer Learning Models (Table 5 & Figure 9)

| Model Architecture | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | Hamming Loss (%) | MCC (%) | JS (%) | CK (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ⭐ **Proposed TumorGANet (VGG-16)** | **99.53** | **100.0** | **100.0** | **99.00** | **0.20** | **99.50** | **98.50** | **99.50** |
| GANs + Inception-v3 | 99.34 | 99.34 | 99.34 | 99.00 | 1.02 | 97.20 | 95.80 | 97.20 |
| GANs + ResNet-101 | 98.93 | 98.93 | 98.93 | 99.00 | 0.50 | 99.10 | 97.90 | 98.70 |
| GANs + AlexNet | 98.17 | 98.17 | 98.17 | 98.00 | 0.90 | 97.55 | 96.20 | 97.50 |
| GANs + GoogLeNet | 97.18 | 97.18 | 97.18 | 97.00 | 2.44 | 94.10 | 90.10 | 93.20 |
| GANs + VGG-19 | 96.72 | 96.72 | 96.72 | 96.50 | 1.60 | 96.00 | 93.10 | 96.00 |
| GANs + MobileNet | 96.41 | 96.41 | 96.41 | 96.20 | 7.90 | 95.00 | 93.00 | 95.20 |
| GANs + MobileNet-v2 | 95.85 | 95.85 | 95.85 | 95.50 | 2.44 | 94.10 | 90.20 | 93.10 |
| GANs + DenseNet-121 | 95.78 | 95.78 | 95.78 | 95.60 | 2.47 | 94.10 | 90.10 | 93.10 |
| GANs + Xception | 78.01 | 86.00 | 77.00 | 76.00 | 10.90 | 64.30 | 64.30 | 70.70 |

### 2. Class-wise Performance for Proposed TumorGANet (Table 6)

| Tumor Types | Precision (%) | Recall (%) | F1-score (%) |
| :--- | :---: | :---: | :---: |
| **Glioma** | 99.00 | 97.00 | 98.00 |
| **Meningioma** | 97.00 | 99.00 | 98.00 |
| **No Tumor** | 100.0 | 100.0 | 100.0 |
| **Pituitary** | 100.0 | 99.00 | 99.00 |

### 3. Ablation Study of Optimizer & Learning Rate (Table 7)

| Optimizer | Learning Rate | Epochs | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | HL (%) | MCC (%) | JS (%) | CK (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Adam (Proposed)** | **0.0001** | **50** | **98.00** | **99.00** | **100.0** | **99.00** | **2.20** | **96.00** | **95.00** | **96.10** |
| Adam | 0.001 | 50 | 95.00 | 100.0 | 99.00 | 98.00 | 4.70 | 93.70 | 91.00 | 93.60 |
| Adam | 0.00001 | 50 | 99.00 | 100.0 | 100.0 | 100.0 | 0.60 | 99.10 | 98.70 | 99.10 |
| SGD | 0.001 | 50 | 99.00 | 100.0 | 100.0 | 100.0 | 0.40 | 99.30 | 99.00 | 99.30 |
| SGD | 0.0001 | 50 | 99.00 | 100.0 | 100.0 | 100.0 | 1.30 | 98.10 | 97.30 | 98.10 |
| SGD | 0.00001 | 50 | 88.00 | 94.00 | 100.0 | 94.00 | 11.70 | 94.30 | 79.40 | 84.20 |
| Adagrad | 0.001 | 50 | 98.00 | 99.00 | 100.0 | 99.00 | 1.70 | 97.60 | 96.50 | 97.60 |
| Adagrad | 0.0001 | 50 | 88.00 | 97.00 | 100.0 | 96.00 | 11.80 | 84.40 | 78.90 | 84.00 |
| Adagrad | 0.00001 | 50 | 73.00 | 81.00 | 93.00 | 96.00 | 27.00 | 64.40 | 57.10 | 63.50 |
| Nadam | 0.001 | 50 | 98.00 | 100.0 | 100.0 | 99.00 | 1.60 | 95.50 | 97.40 | 97.40 |
| Nadam | 0.0001 | 50 | 98.00 | 100.0 | 100.0 | 99.00 | 1.80 | 97.50 | 96.40 | 97.50 |
| Nadam | 0.00001 | 50 | 99.00 | 100.0 | 100.0 | 100.0 | 1.00 | 98.50 | 97.90 | 98.50 |
| RMSprop | 0.001 | 50 | 92.00 | 100.0 | 99.00 | 94.00 | 7.90 | 98.50 | 85.60 | 85.30 |
| RMSprop | 0.0001 | 50 | 92.00 | 100.0 | 99.00 | 94.00 | 7.70 | 90.90 | 85.40 | 89.60 |
| RMSprop | 0.00001 | 50 | 99.00 | 100.0 | 100.0 | 100.0 | 0.80 | 98.80 | 98.30 | 98.80 |

---

## 🔍 Explainable AI (LIME)

To ensure clinical trustworthiness and eliminate black-box opacity, the system implements **LIME** (Equation 21) replicating **Figure 11**:
1. **Original Image**: Raw axial MRI scan (224x224x3).
2. **Correct Region**: Super-pixel segmentation with background suppression, isolating the lesion mass.
3. **Highlight Contour**: High-contrast yellow demarcation contour delineating the tumor border.
4. **Heatmap Overlay**: Transparent attribution overlay (green indicates positive predictive contribution; red indicates negative contribution).

---

## 📚 Citation

If you find this work or implementation useful, please cite the base paper:

```bibtex
@ARTICLE{nag2024tumorganet,
  author={Nag, Anindya and Mondal, Hirak and Hassan, Md Mehedi and Al-Shehari, Taher and Kadrie, Mohammed and Al-Razgan, Muna and Alfakih, Taha and Biswas, Sujit and Bairagi, Anupam Kumar},
  journal={IEEE Access}, 
  title={TumorGANet: A Transfer Learning and Generative Adversarial Network-Based Data Augmentation Model for Brain Tumor Classification}, 
  year={2024},
  volume={12},
  pages={103060-103081},
  doi={10.1109/ACCESS.2024.3429633}
}
```

