"""
Ablation Study Engine for TumorGANet.
Implements Section VII and reproduces Table 7 of the paper:
Evaluates 5 optimizers (Adam, SGD, Adagrad, Nadam, RMSprop)
across 3 learning rates (0.001, 0.0001, 0.00001) with 50 epochs.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from data.dataset import get_data_loaders
from training.train_classifier import TumorGANetTrainer


# Published Ablation Benchmark Data from Table 7
PUBLISHED_TABLE_7 = [
    {"Optimizer": "Adam", "Learning Rate": 0.001, "Epoch": 50, "Precision (%)": 100.0, "Recall (%)": 99.00, "F1-score (%)": 98.00, "Accuracy (%)": 95.00, "HL (%)": 4.70, "MCC (%)": 93.70, "JS (%)": 91.00, "CK (%)": 93.60},
    {"Optimizer": "Adam", "Learning Rate": 0.0001, "Epoch": 50, "Precision (%)": 99.00, "Recall (%)": 100.0, "F1-score (%)": 99.00, "Accuracy (%)": 98.00, "HL (%)": 2.20, "MCC (%)": 96.00, "JS (%)": 95.00, "CK (%)": 96.10},
    {"Optimizer": "Adam", "Learning Rate": 0.00001, "Epoch": 50, "Precision (%)": 100.0, "Recall (%)": 100.0, "F1-score (%)": 100.0, "Accuracy (%)": 99.00, "HL (%)": 0.60, "MCC (%)": 99.10, "JS (%)": 98.70, "CK (%)": 99.10},
    {"Optimizer": "SGD", "Learning Rate": 0.001, "Epoch": 50, "Precision (%)": 100.0, "Recall (%)": 100.0, "F1-score (%)": 100.0, "Accuracy (%)": 99.00, "HL (%)": 0.40, "MCC (%)": 99.30, "JS (%)": 99.00, "CK (%)": 99.30},
    {"Optimizer": "SGD", "Learning Rate": 0.0001, "Epoch": 50, "Precision (%)": 100.0, "Recall (%)": 100.0, "F1-score (%)": 100.0, "Accuracy (%)": 99.00, "HL (%)": 1.30, "MCC (%)": 98.10, "JS (%)": 97.30, "CK (%)": 98.10},
    {"Optimizer": "SGD", "Learning Rate": 0.00001, "Epoch": 50, "Precision (%)": 94.00, "Recall (%)": 100.0, "F1-score (%)": 94.00, "Accuracy (%)": 88.00, "HL (%)": 11.70, "MCC (%)": 94.30, "JS (%)": 79.40, "CK (%)": 84.20},
    {"Optimizer": "Adagrad", "Learning Rate": 0.001, "Epoch": 50, "Precision (%)": 99.00, "Recall (%)": 100.0, "F1-score (%)": 99.00, "Accuracy (%)": 98.00, "HL (%)": 1.70, "MCC (%)": 97.60, "JS (%)": 96.50, "CK (%)": 97.60},
    {"Optimizer": "Adagrad", "Learning Rate": 0.0001, "Epoch": 50, "Precision (%)": 97.00, "Recall (%)": 100.0, "F1-score (%)": 96.00, "Accuracy (%)": 88.00, "HL (%)": 11.80, "MCC (%)": 84.40, "JS (%)": 78.90, "CK (%)": 84.00},
    {"Optimizer": "Adagrad", "Learning Rate": 0.00001, "Epoch": 50, "Precision (%)": 81.00, "Recall (%)": 93.00, "F1-score (%)": 96.00, "Accuracy (%)": 73.00, "HL (%)": 27.00, "MCC (%)": 64.40, "JS (%)": 57.10, "CK (%)": 63.50},
    {"Optimizer": "Nadam", "Learning Rate": 0.001, "Epoch": 50, "Precision (%)": 100.0, "Recall (%)": 100.0, "F1-score (%)": 99.00, "Accuracy (%)": 98.00, "HL (%)": 1.60, "MCC (%)": 95.50, "JS (%)": 97.40, "CK (%)": 97.40},
    {"Optimizer": "Nadam", "Learning Rate": 0.0001, "Epoch": 50, "Precision (%)": 100.0, "Recall (%)": 100.0, "F1-score (%)": 99.00, "Accuracy (%)": 98.00, "HL (%)": 1.80, "MCC (%)": 97.50, "JS (%)": 96.40, "CK (%)": 97.50},
    {"Optimizer": "Nadam", "Learning Rate": 0.00001, "Epoch": 50, "Precision (%)": 100.0, "Recall (%)": 100.0, "F1-score (%)": 100.0, "Accuracy (%)": 99.00, "HL (%)": 1.00, "MCC (%)": 98.50, "JS (%)": 97.90, "CK (%)": 98.50},
    {"Optimizer": "RMSprop", "Learning Rate": 0.001, "Epoch": 50, "Precision (%)": 100.0, "Recall (%)": 99.00, "F1-score (%)": 94.00, "Accuracy (%)": 92.00, "HL (%)": 7.90, "MCC (%)": 98.50, "JS (%)": 85.60, "CK (%)": 85.30},
    {"Optimizer": "RMSprop", "Learning Rate": 0.0001, "Epoch": 50, "Precision (%)": 100.0, "Recall (%)": 99.00, "F1-score (%)": 94.00, "Accuracy (%)": 92.00, "HL (%)": 7.70, "MCC (%)": 90.90, "JS (%)": 85.40, "CK (%)": 89.60},
    {"Optimizer": "RMSprop", "Learning Rate": 0.00001, "Epoch": 50, "Precision (%)": 100.0, "Recall (%)": 100.0, "F1-score (%)": 100.0, "Accuracy (%)": 99.00, "HL (%)": 0.80, "MCC (%)": 98.80, "JS (%)": 98.30, "CK (%)": 98.80},
]


def get_published_ablation_df() -> pd.DataFrame:
    """Returns Table 7 as a formatted pandas DataFrame."""
    return pd.DataFrame(PUBLISHED_TABLE_7)


def export_ablation_tables(output_dir: Path = config.RESULTS_DIR):
    """Exports Table 7 to CSV and Markdown."""
    output_dir = Path(output_dir)
    df = get_published_ablation_df()
    df.to_csv(output_dir / "table_7_ablation_study.csv", index=False)
    with open(output_dir / "table_7_ablation_study.md", "w", encoding="utf-8") as f:
        f.write("# Table 7: Ablation Study of TumorGANet Model (IEEE Access 2024)\n\n")
        f.write(df.to_markdown(index=False))
    print(f"Table 7 ablation study successfully exported to {output_dir}")


def run_ablation_experiment(
    data_dir: Path = config.RAW_DATA_DIR,
    epochs_per_config: int = 3,
    output_csv: Path = config.RESULTS_DIR / "custom_ablation_results.csv"
) -> pd.DataFrame:
    """
    Executes a custom ablation grid across all 5 optimizers and 3 learning rates.
    """
    train_loader, test_loader = get_data_loaders(data_dir, batch_size=config.MODEL_CONFIG["batch_size"])
    results = []

    print("\n" + "=" * 80)
    print(" Running Full Ablation Study Grid (5 Optimizers x 3 Learning Rates)")
    print("=" * 80)

    for opt in config.ABLATION_OPTIMIZERS:
        for lr in config.ABLATION_LEARNING_RATES:
            print(f"\n--> Evaluating Configuration: Optimizer = {opt} | Learning Rate = {lr}")
            trainer = TumorGANetTrainer(learning_rate=lr, optimizer_name=opt)
            trainer.fit(train_loader, test_loader, epochs=epochs_per_config)
            metrics = trainer.evaluate(test_loader)["summary"]

            results.append({
                "Optimizer": opt,
                "Learning Rate": lr,
                "Epoch": epochs_per_config,
                "Precision (%)": round(metrics["precision"], 2),
                "Recall (%)": round(metrics["recall"], 2),
                "F1-score (%)": round(metrics["f1_score"], 2),
                "Accuracy (%)": round(metrics["accuracy"], 2),
                "HL (%)": round(metrics["hamming_loss_percent"], 2),
                "MCC (%)": round(metrics["matthews_corrcoef"], 2),
                "JS (%)": round(metrics["jaccard_score"], 2),
                "CK (%)": round(metrics["cohens_kappa"], 2)
            })

    res_df = pd.DataFrame(results)
    res_df.to_csv(output_csv, index=False)
    print(f"\nAblation study completed! Results saved to {output_csv}")
    return res_df


if __name__ == "__main__":
    export_ablation_tables()

