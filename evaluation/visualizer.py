"""
Visualization Module for TumorGANet.
Reproduces all figures and visual analyses presented in the published paper:
- Figure 6: GAN Training History (Loss curves) & KDE Density Estimation
- Figure 7: Training and Validation Metrics (Loss & Accuracy) over Epochs
- Figure 8: Confusion Matrices for All 10 Transfer Learning Models
- Figure 9: Comparison Bar Charts (Precision, Recall, F1-Score, Accuracy)
- Figure 10: Comparison Line Chart (Testing Loss, Validation Loss, Train/Val/Test Accuracy)
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


class TumorGANetVisualizer:
    """
    Publication-grade plotting toolkit reproducing Figures 6, 7, 8, 9, 10 of the paper.
    """

    def __init__(self, output_dir: Path = config.RESULTS_DIR):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Set clinical / paper styling
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        model_name: str = "TumorGANet (GANs + VGG-16)",
        classes: List[str] = config.CLASSES,
        save_filename: Optional[str] = None
    ) -> plt.Figure:
        """
        Plots confusion matrix matching Figure 8 of the paper.
        """
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=[c.capitalize() for c in classes],
            yticklabels=[c.capitalize() for c in classes],
            cbar=True,
            ax=ax
        )
        ax.set_title(f"Confusion Matrix\n{model_name}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted Label", fontsize=11)
        ax.set_ylabel("True Label", fontsize=11)
        plt.tight_layout()

        if save_filename:
            save_path = self.output_dir / save_filename
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig

    def plot_training_history(
        self,
        history: Dict[str, List[float]],
        model_name: str = "Proposed TumorGANet (GANs + VGG-16)",
        save_filename: Optional[str] = None
    ) -> plt.Figure:
        """
        Plots Training and Validation Accuracy & Loss over Epochs matching Figure 7.
        """
        epochs = range(1, len(history["train_loss"]) + 1)
        fig, ax = plt.subplots(figsize=(7, 5))

        ax.plot(epochs, history.get("val_loss", []), label="Validation Loss", color="#1f77b4", linewidth=1.8)
        ax.plot(epochs, history.get("val_acc", []), label="Validation Accuracy", color="#ff7f0e", linewidth=1.8)
        ax.plot(epochs, history.get("train_acc", []), label="Train Accuracy", color="#2ca02c", linewidth=1.8)
        ax.plot(epochs, history.get("train_loss", []), label="Train Loss", color="#d62728", linewidth=1.8)

        ax.set_title(f"Training and Validation Metrics over Epochs\n{model_name}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Epochs", fontsize=10)
        ax.set_ylabel("Metric Value", fontsize=10)
        ax.legend(loc="center right", frameon=True, fontsize=9)
        plt.tight_layout()

        if save_filename:
            save_path = self.output_dir / save_filename
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig

    def plot_gan_history(
        self,
        g_losses: List[float],
        d_losses: List[float],
        save_filename: Optional[str] = "figure_6_gan_loss.png"
    ) -> plt.Figure:
        """
        Plots GAN training history matching Figure 6 of the paper.
        """
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(d_losses, label="Discriminator Loss", color="#1f77b4", alpha=0.85, linewidth=1.2)
        ax.plot(g_losses, label="Generator Loss", color="#ff7f0e", alpha=0.85, linewidth=1.2)
        ax.set_title("GANs Training History", fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch / Iteration", fontsize=11)
        ax.set_ylabel("Loss", fontsize=11)
        ax.legend(loc="upper right", frameon=True)
        plt.tight_layout()

        if save_filename:
            save_path = self.output_dir / save_filename
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig

    def plot_kde_comparison(
        self,
        real_pixels: np.ndarray,
        synth_pixels: np.ndarray,
        save_filename: Optional[str] = "kde_density_comparison.png"
    ) -> plt.Figure:
        """
        Kernel Density Estimation (KDE) comparing real vs synthetic MRI pixel distributions (Section V-B).
        """
        fig, ax = plt.subplots(figsize=(7, 4))

        sns.kdeplot(real_pixels.flatten(), ax=ax, label="Actual Real MRI", color="#1f77b4", fill=True, alpha=0.3)
        sns.kdeplot(synth_pixels.flatten(), ax=ax, label="Generated Synthetic MRI", color="#ff7f0e", fill=True, alpha=0.3)

        ax.set_title("Kernel Density Estimation (KDE): Real vs Generated MRI", fontsize=12, fontweight="bold")
        ax.set_xlabel("Normalized Pixel Intensity", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.legend(loc="upper right")
        plt.tight_layout()

        if save_filename:
            save_path = self.output_dir / save_filename
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig

    def plot_figure_9_benchmark_bars(
        self,
        comparison_dict: Optional[Dict[str, Dict[str, float]]] = None,
        save_filename: Optional[str] = "figure_9_model_comparison_bars.png"
    ) -> plt.Figure:
        """
        Generates grouped bar chart comparing all 10 transfer learning models by
        Precision, Recall, F1-Score, and Accuracy (Figure 9 in paper).
        """
        # Default benchmark data from Table 5 & Figure 9 if not provided
        if comparison_dict is None:
            models_data = {
                "GoogLeNet": {"Precision": 97.18, "Recall": 97.18, "F1-score": 97.0, "Accuracy": 97.18},
                "MobileNet-v2": {"Precision": 95.85, "Recall": 95.85, "F1-score": 95.5, "Accuracy": 95.85},
                "DenseNet-121": {"Precision": 95.78, "Recall": 95.78, "F1-score": 95.6, "Accuracy": 95.78},
                "MobileNet": {"Precision": 96.41, "Recall": 96.41, "F1-score": 96.2, "Accuracy": 96.41},
                "VGG-19": {"Precision": 96.72, "Recall": 96.72, "F1-score": 96.5, "Accuracy": 96.72},
                "AlexNet": {"Precision": 98.17, "Recall": 98.17, "F1-score": 98.0, "Accuracy": 98.17},
                "Xception": {"Precision": 86.00, "Recall": 77.00, "F1-score": 76.0, "Accuracy": 78.01},
                "ResNet-101": {"Precision": 98.93, "Recall": 98.93, "F1-score": 99.0, "Accuracy": 98.93},
                "Inception-v3": {"Precision": 99.34, "Recall": 99.34, "F1-score": 99.0, "Accuracy": 99.34},
                "TumorGANet\n(VGG-16)": {"Precision": 100.0, "Recall": 100.0, "F1-score": 99.0, "Accuracy": 99.53}
            }
        else:
            models_data = comparison_dict

        model_names = list(models_data.keys())
        x = np.arange(len(model_names))
        width = 0.20

        fig, ax = plt.subplots(figsize=(14, 6))

        prec = [models_data[m]["Precision"] for m in model_names]
        rec = [models_data[m]["Recall"] for m in model_names]
        f1 = [models_data[m]["F1-score"] for m in model_names]
        acc = [models_data[m]["Accuracy"] for m in model_names]

        ax.bar(x - 1.5 * width, prec, width, label="Precision (%)", color="#4682B4")
        ax.bar(x - 0.5 * width, rec, width, label="Recall (%)", color="#87CEEB")
        ax.bar(x + 0.5 * width, f1, width, label="F1-score (%)", color="#1E3F66")
        ax.bar(x + 1.5 * width, acc, width, label="Accuracy (%)", color="#000080")

        ax.set_ylabel("Value (%)", fontsize=11, fontweight="bold")
        ax.set_title("Comparison of Model Performance of All Applied Models (Figure 9)", fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=25, ha="right", fontsize=9)
        ax.set_ylim(0, 110)
        ax.legend(loc="lower left", frameon=True)
        plt.tight_layout()

        if save_filename:
            save_path = self.output_dir / save_filename
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig

