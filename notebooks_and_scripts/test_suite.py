"""
Comprehensive Test Suite for TumorGANet.
Validates:
1. Mathematical Preprocessing (Equations 2-7)
2. TumorGANet VGG-16 Architecture & Forward Pass
3. DCGAN Generator & Discriminator Dimensions (Equation 8)
4. Evaluation Metrics Calculation (Equations 13-20: Accuracy, HL, MCC, JS, CK)
5. LIME Explainability Generation (Figure 11)
6. Sample Brain MRI Generation
"""

import unittest
from pathlib import Path
import numpy as np
from PIL import Image
import torch

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from data.preprocessor import BrainMRIPreprocessor
from data.dataset import create_synthetic_mri_slice
from models.tumor_ganet import TumorGANet
from models.gan import Generator, Discriminator
from evaluation.metrics import compute_all_metrics
from evaluation.explainability import BrainMRILIMEExplainer


class TestTumorGANetSystem(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cpu")
        self.preprocessor = BrainMRIPreprocessor(is_training=True)

    def test_01_preprocessing_math(self):
        """Validates resizing, normalization, and augmentation formulas (Equations 2-7)."""
        dummy_img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)

        # 1. Resizing (Equations 2-3)
        resized = self.preprocessor.resize_image(dummy_img, target_size=(224, 224))
        self.assertEqual(resized.shape, (224, 224, 3))

        # 2. Normalization (Equations 4-5)
        normalized = self.preprocessor.normalize_pixels(resized, a=1.0, b=0.0)
        self.assertTrue(normalized.min() >= 0.0)
        self.assertTrue(normalized.max() <= 1.0)

        # 3. Zooming (Equation 6)
        zoomed = self.preprocessor.random_zoom(resized, zoom_range=0.2)
        self.assertEqual(zoomed.shape, (224, 224, 3))

        # 4. Horizontal Flip (Equation 7)
        flipped = self.preprocessor.horizontal_flip_image(resized)
        self.assertEqual(flipped.shape, (224, 224, 3))
        # Ensure flipping reverses width
        np.testing.assert_array_equal(flipped[:, 0, :], resized[:, -1, :])

    def test_02_tumorganet_architecture(self):
        """Validates TumorGANet model forward pass and custom 1024-dense classification head."""
        model = TumorGANet(num_classes=4, pretrained=False).to(self.device)
        model.eval()

        dummy_batch = torch.randn(2, 3, 224, 224, device=self.device)
        with torch.no_grad():
            logits = model(dummy_batch)
            probs = model.predict_proba(dummy_batch)

        self.assertEqual(logits.shape, (2, 4))
        self.assertEqual(probs.shape, (2, 4))
        # Probabilities sum to 1
        sums = probs.sum(dim=1).numpy()
        np.testing.assert_allclose(sums, np.ones(2), rtol=1e-5)

    def test_03_dcgan_dimensions(self):
        """Validates Generator and Discriminator tensor dimensions (Algorithm 1 & Section V-B)."""
        latent_dim = config.GAN_CONFIG["latent_dim"]
        netG = Generator(latent_dim=latent_dim).to(self.device)
        netD = Discriminator().to(self.device)

        z = torch.randn(4, latent_dim, device=self.device)
        fake_mri = netG(z)
        self.assertEqual(fake_mri.shape, (4, 3, 224, 224))

        validity = netD(fake_mri)
        self.assertEqual(validity.shape, (4, 1))
        # Validity in range [0, 1] due to Sigmoid
        self.assertTrue((validity >= 0.0).all().item())
        self.assertTrue((validity <= 1.0).all().item())

    def test_04_medical_evaluation_metrics(self):
        """Validates Accuracy, Hamming Loss, MCC, Jaccard Score, and Cohen's Kappa (Equations 13-20)."""
        # Simulated test predictions: 95% agreement
        y_true = [0, 0, 1, 1, 2, 2, 3, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
        y_pred = [0, 0, 1, 1, 2, 2, 3, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 2] # 1 error out of 20 = 95% accuracy

        results = compute_all_metrics(y_true, y_pred)
        summary = results["summary"]

        self.assertEqual(summary["accuracy"], 95.0)
        # Equation 17: Hamming Loss = 1/20 = 0.05 = 5.0%
        self.assertAlmostEqual(summary["hamming_loss_percent"], 5.0, places=2)
        # Check MCC and Cohen's Kappa are high
        self.assertGreater(summary["matthews_corrcoef"], 90.0)
        self.assertGreater(summary["cohens_kappa"], 90.0)
        self.assertGreater(summary["jaccard_score"], 85.0)

    def test_05_synthetic_mri_generation(self):
        """Validates synthetic MRI generator for all four classes."""
        for cname in config.CLASSES:
            mri_img = create_synthetic_mri_slice(cname, size=(224, 224))
            self.assertEqual(mri_img.size, (224, 224))
            self.assertEqual(mri_img.mode, "RGB")

    def test_06_lime_explainer(self):
        """Validates LIME explainability pipeline generating Figure 11 panels."""
        model = TumorGANet(num_classes=4, pretrained=False).to(self.device)
        explainer = BrainMRILIMEExplainer(model=model, device=self.device)

        test_img = create_synthetic_mri_slice("glioma")
        explanation = explainer.explain_image(test_img, num_samples=20, num_features=3)

        self.assertIn("original_image", explanation)
        self.assertIn("correct_region", explanation)
        self.assertIn("highlight_contour", explanation)
        self.assertIn("heatmap", explanation)
        self.assertEqual(explanation["original_image"].shape, (224, 224, 3))
        self.assertEqual(explanation["correct_region"].shape, (224, 224, 3))
        self.assertEqual(explanation["highlight_contour"].shape, (224, 224, 3))
        self.assertEqual(explanation["heatmap"].shape, (224, 224, 3))


if __name__ == "__main__":
    unittest.main()

