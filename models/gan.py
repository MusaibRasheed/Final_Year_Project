"""
Generative Adversarial Network (GAN) for Brain Tumor MRI Data Augmentation.
Implements the architecture and training objectives from Section V-B, Equation (8),
and Algorithm 1 of the TumorGANet paper.
"""

import os
from pathlib import Path
from typing import Tuple, Optional
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


def weights_init_normal(m):
    """Initializes convolutional and batchnorm layers with normal distribution."""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    """
    DCGAN Generator Architecture for synthesizing 224x224 (or 112x112/64x64) MRI scans.
    Takes latent noise vector z ~ N(0, 1) and upsamples to synthetic MRI slices.
    """

    def __init__(self, latent_dim: int = config.GAN_CONFIG["latent_dim"], channels: int = 3):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.init_size = 7  # 7 -> 14 -> 28 -> 56 -> 112 -> 224
        self.l1 = nn.Sequential(
            nn.Linear(latent_dim, 512 * self.init_size * self.init_size),
            nn.BatchNorm1d(512 * self.init_size * self.init_size),
            nn.ReLU(True)
        )

        self.conv_blocks = nn.Sequential(
            # 7x7 -> 14x14
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            # 14x14 -> 28x28
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # 28x28 -> 56x56
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # 56x56 -> 112x112
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            # 112x112 -> 224x224
            nn.ConvTranspose2d(32, channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()  # Produces output in [-1, 1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.l1(z)
        out = out.view(out.shape[0], 512, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    """
    DCGAN Discriminator Architecture for classifying real vs synthetic MRI scans.
    Implements strided convolutions with LeakyReLU and Dropout.
    """

    def __init__(self, channels: int = 3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True, stride=2):
            block = [nn.Conv2d(in_filters, out_filters, 4, stride, 1, bias=False)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters))
            block.append(nn.LeakyReLU(0.2, inplace=True))
            block.append(nn.Dropout2d(0.25))
            return block

        self.model = nn.Sequential(
            # 224x224 -> 112x112
            *discriminator_block(channels, 32, bn=False),
            # 112x112 -> 56x56
            *discriminator_block(32, 64),
            # 56x56 -> 28x28
            *discriminator_block(64, 128),
            # 28x28 -> 14x14
            *discriminator_block(128, 256),
            # 14x14 -> 7x7
            *discriminator_block(256, 512),
        )

        # Output probability scalar: D(x) in [0, 1]
        self.adv_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 1),
            nn.Sigmoid()
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        features = self.model(img)
        validity = self.adv_layer(features)
        return validity


class BrainMRIGAN:
    """
    Controller class for training, generating, and saving GAN models for brain tumor MRI augmentation.
    """

    def __init__(
        self,
        latent_dim: int = config.GAN_CONFIG["latent_dim"],
        lr_g: float = config.GAN_CONFIG["lr_g"],
        lr_d: float = config.GAN_CONFIG["lr_d"],
        device: Optional[str] = None
    ):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.latent_dim = latent_dim

        self.generator = Generator(latent_dim=latent_dim).to(self.device)
        self.discriminator = Discriminator().to(self.device)

        self.generator.apply(weights_init_normal)
        self.discriminator.apply(weights_init_normal)

        self.adversarial_loss = nn.BCELoss()

        self.optimizer_G = torch.optim.Adam(
            self.generator.parameters(),
            lr=lr_g,
            betas=(config.GAN_CONFIG["beta1"], config.GAN_CONFIG["beta2"])
        )
        self.optimizer_D = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=lr_d,
            betas=(config.GAN_CONFIG["beta1"], config.GAN_CONFIG["beta2"])
        )

    def train_step(self, real_imgs: torch.Tensor) -> Tuple[float, float]:
        """
        Executes a single GAN step following Algorithm 1:
        1. Train Discriminator with half real and half fake images.
        2. Train Generator with generated images and real labels.
        """
        batch_size = real_imgs.size(0)
        real_imgs = real_imgs.to(self.device)

        # Ground truth labels
        valid = torch.ones(batch_size, 1, device=self.device, requires_grad=False)
        fake = torch.zeros(batch_size, 1, device=self.device, requires_grad=False)

        # ---------------------
        #  Train Discriminator
        # ---------------------
        self.optimizer_D.zero_grad()

        # Real images loss
        real_pred = self.discriminator(real_imgs)
        d_real_loss = self.adversarial_loss(real_pred, valid)

        # Fake images loss
        z = torch.randn(batch_size, self.latent_dim, device=self.device)
        gen_imgs = self.generator(z)
        fake_pred = self.discriminator(gen_imgs.detach())
        d_fake_loss = self.adversarial_loss(fake_pred, fake)

        # Combined discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2
        d_loss.backward()
        self.optimizer_D.step()

        # -----------------
        #  Train Generator
        # -----------------
        self.optimizer_G.zero_grad()

        # Generate fresh images & target real labels
        z = torch.randn(batch_size, self.latent_dim, device=self.device)
        gen_imgs = self.generator(z)
        pred_valid = self.discriminator(gen_imgs)
        g_loss = self.adversarial_loss(pred_valid, valid)

        g_loss.backward()
        self.optimizer_G.step()

        return g_loss.item(), d_loss.item()

    def generate_synthetic_samples(self, num_samples: int = 16) -> np.ndarray:
        """
        Generates synthetic MRI images as numpy array (N, 224, 224, 3) in range [0, 255].
        """
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim, device=self.device)
            gen_imgs = self.generator(z)
            # Map [-1, 1] -> [0, 1] -> [0, 255]
            gen_imgs = (gen_imgs * 0.5 + 0.5).clamp(0, 1)
            gen_imgs = gen_imgs.cpu().permute(0, 2, 3, 1).numpy()
            images_uint8 = (gen_imgs * 255).astype(np.uint8)
        self.generator.train()
        return images_uint8

    def save_models(self, checkpoint_dir: Path = config.CHECKPOINTS_DIR):
        """Saves generator and discriminator weights."""
        checkpoint_dir = Path(checkpoint_dir)
        torch.save(self.generator.state_dict(), checkpoint_dir / "generator_best.pth")
        torch.save(self.discriminator.state_dict(), checkpoint_dir / "discriminator_best.pth")

    def load_models(self, checkpoint_dir: Path = config.CHECKPOINTS_DIR):
        """Loads generator and discriminator weights if present."""
        checkpoint_dir = Path(checkpoint_dir)
        g_path = checkpoint_dir / "generator_best.pth"
        d_path = checkpoint_dir / "discriminator_best.pth"
        if g_path.exists():
            self.generator.load_state_dict(torch.load(g_path, map_location=self.device))
        if d_path.exists():
            self.discriminator.load_state_dict(torch.load(d_path, map_location=self.device))

