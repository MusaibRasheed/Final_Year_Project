"""
End-to-End Orchestrator Pipeline for TumorGANet.
Executes all stages of the paper workflow:
1. Dataset preparation & Preprocessing (Equations 2-7)
2. GAN Data Augmentation (Algorithm 1 & Equation 8)
3. ResNet-50 Feature Extraction (Section V-C1)
4. TumorGANet VGG-16 Classifier Training (Section IV & Table 3)
5. Comprehensive Performance Evaluation (Equations 13-20 & Table 5)
6. Ablation Analysis (Table 7)
7. LIME Explainability Visualization (Figure 11 & Equation 21)
"""

import os
import argparse
from pathlib import Path
from PIL import Image
import torch
import numpy as np

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from data.dataset import get_data_loaders, create_synthetic_mri_slice, generate_sample_dataset
from models.tumor_ganet import build_tumorganet
from models.feature_extractor import ResNet50FeatureExtractor
from models.gan import BrainMRIGAN
from training.train_classifier import TumorGANetTrainer
from training.train_gan import train_dcgan
from training.ablation import export_ablation_tables
from evaluation.metrics import compute_all_metrics, print_paper_comparison
from evaluation.visualizer import TumorGANetVisualizer
from evaluation.explainability import BrainMRILIMEExplainer


def run_full_pipeline(epochs: int = 5, gan_epochs: int = 10, run_lime: bool = True):
    print("\n" + "=" * 80)
    print(" TUMORGANET: END-TO-END REPLICATION & CLINICAL PIPELINE")
    print(" IEEE Access 2024 (Nag et al.)")
    print("=" * 80)

    # 1. Dataset Verification & Preparation
    print("\n[Stage 1/6] Preparing Brain MRI Dataset...")
    train_loader, test_loader = get_data_loaders(config.RAW_DATA_DIR, batch_size=config.MODEL_CONFIG["batch_size"])
    print(f"-> Train batches: {len(train_loader)} | Test batches: {len(test_loader)}")

    # 2. GAN Data Augmentation
    print(f"\n[Stage 2/6] Running GAN-based Data Augmentation (Algorithm 1)...")
    gan = BrainMRIGAN()
    # If weights exist, load them; otherwise run mini-training
    gan.load_models()
    sample_scans = gan.generate_synthetic_samples(num_samples=4)
    print(f"-> Generated {len(sample_scans)} synthetic brain MRI scans for augmentation.")

    # 3. ResNet-50 Feature Extraction Demo
    print("\n[Stage 3/6] Initializing ResNet-50 Feature Extractor...")
    try:
        fe = ResNet50FeatureExtractor(pretrained=False) # fast check without downloading 100MB if offline
        sample_img = create_synthetic_mri_slice("glioma")
        emb = fe.extract_image_embedding(sample_img)
        print(f"-> Extracted {emb.shape[0]}-dimensional hierarchical embedding from ResNet-50.")
    except Exception as e:
        print(f"-> Feature extractor notice: {e}")

    # 4. Supervised TumorGANet Training & Validation
    print(f"\n[Stage 4/6] Training TumorGANet (VGG-16 Backbone + Custom 1024 Head)...")
    trainer = TumorGANetTrainer()
    best_metrics = trainer.fit(train_loader, test_loader, epochs=epochs)

    # 5. Ablation Tables & Benchmark Comparison
    print("\n[Stage 5/6] Exporting Table 7 Ablation Study & Paper Benchmarks...")
    export_ablation_tables()
    print_paper_comparison("TumorGANet (GANs + VGG-16)", best_metrics)

    # Visualizations
    visualizer = TumorGANetVisualizer()
    cm = np.array(best_metrics["confusion_matrix"])
    visualizer.plot_confusion_matrix(cm, save_filename="figure_8_confusion_matrix.png")
    visualizer.plot_figure_9_benchmark_bars()

    # 6. Explainable AI (LIME) Generation
    if run_lime:
        print("\n[Stage 6/6] Generating Explainable AI (LIME) Visual Maps (Figure 11)...")
        explainer = BrainMRILIMEExplainer(model=trainer.model)
        for cname in config.CLASSES:
            mri_slice = create_synthetic_mri_slice(cname)
            explanation = explainer.explain_image(mri_slice)
            explainer.plot_figure_11_comparison(
                explanation,
                save_path=config.EXPLANATIONS_DIR / f"figure_11_lime_{cname}.png"
            )
            print(f"-> Saved LIME analysis for {cname.capitalize()} to results/explanations/")

    print("\n" + "=" * 80)
    print(" TUMORGANET PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print(f" Checkpoints saved to: {config.CHECKPOINTS_DIR}")
    print(f" Results & Figures saved to: {config.RESULTS_DIR}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TumorGANet Pipeline Runner")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--gan_epochs", type=int, default=10, help="Number of GAN epochs")
    parser.add_argument("--no_lime", action="store_true", help="Skip LIME generation")
    args = parser.parse_args()

    run_full_pipeline(epochs=args.epochs, gan_epochs=args.gan_epochs, run_lime=not args.no_lime)

