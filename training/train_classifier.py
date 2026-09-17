"""
Supervised Training Module for TumorGANet Brain Tumor Classification.
Trains the proposed TumorGANet (VGG-16 Backbone + Custom Dense Classifier)
using the hyperparameters specified in Table 3 of the paper:
- Optimizer: Adam
- Learning Rate: 0.0001 (1e-4)
- Dropout Rate: 0.5
- Dense Units: 1024
- Epochs: 50
- Batch Size: 32
"""

import time
import os
from pathlib import Path
from typing import Dict, Any, Optional
from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from data.dataset import get_data_loaders
from models.tumor_ganet import build_tumorganet, TumorGANet
from evaluation.metrics import compute_all_metrics, print_paper_comparison
from evaluation.visualizer import TumorGANetVisualizer


class TumorGANetTrainer:
    """
    Manages model training, validation, metric logging, and checkpointing.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        learning_rate: float = config.MODEL_CONFIG["learning_rate"],
        optimizer_name: str = config.MODEL_CONFIG["optimizer"],
        device: Optional[str] = None
    ):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model if model is not None else build_tumorganet(pretrained=True, device=str(self.device))
        self.model.to(self.device)

        self.lr = learning_rate
        self.optimizer_name = optimizer_name
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self._build_optimizer(optimizer_name, learning_rate)

        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": []
        }

    def _build_optimizer(self, name: str, lr: float) -> optim.Optimizer:
        params = [p for p in self.model.parameters() if p.requires_grad]
        name = name.lower()
        if name == "adam":
            return optim.Adam(params, lr=lr)
        elif name == "sgd":
            return optim.SGD(params, lr=lr, momentum=0.9)
        elif name == "adagrad":
            return optim.Adagrad(params, lr=lr)
        elif name == "rmsprop":
            return optim.RMSprop(params, lr=lr)
        elif name == "nadam":
            return optim.NAdam(params, lr=lr)
        else:
            return optim.Adam(params, lr=lr)

    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in dataloader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / max(1, total)
        epoch_acc = (correct / max(1, total)) * 100.0
        return epoch_loss, epoch_acc

    def evaluate(self, dataloader: DataLoader) -> Dict[str, Any]:
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())

        total = len(all_labels)
        loss = running_loss / max(1, total)

        metrics = compute_all_metrics(all_labels, all_preds)
        metrics["loss"] = loss
        return metrics

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = config.MODEL_CONFIG["epochs"],
        checkpoint_dir: Path = config.CHECKPOINTS_DIR
    ) -> Dict[str, Any]:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        best_acc = 0.0
        best_metrics = None

        print(f"\nTraining TumorGANet on {self.device} for {epochs} epochs...")
        print(f"Optimizer: {self.optimizer_name} | LR: {self.lr} | Dropout: 0.5")

        for epoch in range(1, epochs + 1):
            start_time = time.time()
            t_loss, t_acc = self.train_epoch(train_loader)
            val_results = self.evaluate(val_loader)

            v_loss = val_results["loss"]
            v_acc = val_results["summary"]["accuracy"]
            duration = time.time() - start_time

            self.history["train_loss"].append(t_loss)
            self.history["train_acc"].append(t_acc / 100.0)
            self.history["val_loss"].append(v_loss)
            self.history["val_acc"].append(v_acc / 100.0)

            print(f"Epoch [{epoch:02d}/{epochs:02d}] ({duration:.1f}s) - "
                  f"Train Loss: {t_loss:.4f}, Train Acc: {t_acc:.2f}% | "
                  f"Val Loss: {v_loss:.4f}, Val Acc: {v_acc:.2f}%")

            if v_acc >= best_acc:
                best_acc = v_acc
                best_metrics = val_results
                torch.save(self.model.state_dict(), checkpoint_dir / "tumorganet_best.pth")

        # Save training history curves matching Figure 7
        visualizer = TumorGANetVisualizer()
        visualizer.plot_training_history(self.history, save_filename="figure_7_training_metrics.png")

        print(f"\nTraining Complete! Best Validation Accuracy: {best_acc:.2f}%")
        return best_metrics


def train_tumorganet_pipeline(
    data_dir: Path = config.RAW_DATA_DIR,
    epochs: int = 5,  # Default to short run for test verification; full paper uses 50
    batch_size: int = config.MODEL_CONFIG["batch_size"]
) -> Dict[str, Any]:
    """
    End-to-end wrapper to load data, fit TumorGANet, and evaluate paper metrics.
    """
    train_loader, test_loader = get_data_loaders(data_dir, batch_size=batch_size)
    trainer = TumorGANetTrainer()
    best_metrics = trainer.fit(train_loader, test_loader, epochs=epochs)

    # Print comparative report against published paper benchmark
    print_paper_comparison("TumorGANet (GANs + VGG-16)", best_metrics)
    return best_metrics

