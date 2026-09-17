"""
GAN Training Script for Brain MRI Data Augmentation.
Implements the training procedure from Section V-B and Algorithm 1 of the TumorGANet paper.
"""

from pathlib import Path
from typing import Optional
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from data.dataset import get_data_loaders
from models.gan import BrainMRIGAN
from evaluation.visualizer import TumorGANetVisualizer


def train_dcgan(
    epochs: int = 50,
    batch_size: int = config.GAN_CONFIG["batch_size"],
    checkpoint_interval: int = 10,
    output_dir: Path = config.SYNTHETIC_DATA_DIR
) -> BrainMRIGAN:
    """
    Trains the DCGAN model to generate synthetic brain MRI scans.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize GAN
    gan = BrainMRIGAN()
    train_loader, _ = get_data_loaders(config.RAW_DATA_DIR, batch_size=batch_size)

    g_loss_history = []
    d_loss_history = []

    print(f"\nStarting GAN Training for {epochs} epochs on {gan.device}...")

    for epoch in range(1, epochs + 1):
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        batches = 0

        for images, _ in train_loader:
            g_l, d_l = gan.train_step(images)
            epoch_g_loss += g_l
            epoch_d_loss += d_l
            batches += 1

        avg_g = epoch_g_loss / max(1, batches)
        avg_d = epoch_d_loss / max(1, batches)
        g_loss_history.append(avg_g)
        d_loss_history.append(avg_d)

        if epoch % 5 == 0 or epoch == 1:
            print(f"GAN Epoch [{epoch:03d}/{epochs:03d}] - D Loss: {avg_d:.4f}, G Loss: {avg_g:.4f}")

        # Save sample visual preview
        if epoch % checkpoint_interval == 0 or epoch == epochs:
            gan.generator.eval()
            with torch.no_grad():
                z = torch.randn(16, config.GAN_CONFIG["latent_dim"], device=gan.device)
                gen_imgs = gan.generator(z)
                gen_imgs = (gen_imgs * 0.5 + 0.5).clamp(0, 1)
                save_image(gen_imgs, output_dir / f"gan_samples_epoch_{epoch:03d}.png", nrow=4)
            gan.generator.train()

    # Save trained weights
    gan.save_models()

    # Plot Figure 6 GAN training loss curve
    visualizer = TumorGANetVisualizer()
    visualizer.plot_gan_history(g_loss_history, d_loss_history)

    print(f"GAN Training complete! Models and samples saved to {output_dir}")
    return gan


if __name__ == "__main__":
    train_dcgan(epochs=5)

