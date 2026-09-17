"""
Evaluation Metrics Module for TumorGANet.
Implements the exact mathematical formulations defined in Section V-F of the paper:
- Accuracy: Equation (13)
- Precision: Equation (14)
- Recall: Equation (15)
- F1-Score: Equation (16)
- Hamming Loss (HL): Equation (17)
- Jaccard Score (JS): Equation (18)
- Matthews Correlation Coefficient (MCC): Equation (19)
- Cohen's Kappa (CK): Equation (20)
"""

from typing import Dict, Any, List, Union, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    hamming_loss,
    jaccard_score,
    matthews_corrcoef,
    cohen_kappa_score,
    confusion_matrix,
    classification_report
)

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


def compute_all_metrics(
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[int], np.ndarray],
    class_names: List[str] = config.CLASSES
) -> Dict[str, Any]:
    """
    Computes all standard and clinical benchmark metrics reported in Table 5 and Table 6.
    Returns metrics formatted both as fractions and percentages matching the paper.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # 1. Overall Aggregate Metrics
    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # Equation (17): Hamming Loss = (1/n) * sum(I[y_i != y_hat_i])
    hl = hamming_loss(y_true, y_pred)

    # Equation (18): Jaccard Score / Index
    js = jaccard_score(y_true, y_pred, average="macro", zero_division=0)

    # Equation (19): Matthews Correlation Coefficient
    mcc = matthews_corrcoef(y_true, y_pred)

    # Equation (20): Cohen's Kappa = (P_o - P_e) / (1 - P_e)
    ck = cohen_kappa_score(y_true, y_pred)

    # 2. Class-wise Metrics (Table 6)
    prec_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    rec_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

    classwise_results = {}
    for idx, cname in enumerate(class_names):
        if idx < len(prec_per_class):
            classwise_results[cname] = {
                "precision": float(prec_per_class[idx] * 100.0),
                "recall": float(rec_per_class[idx] * 100.0),
                "f1_score": float(f1_per_class[idx] * 100.0)
            }

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)

    return {
        "summary": {
            "accuracy": float(acc * 100.0),
            "precision": float(prec_macro * 100.0),
            "recall": float(rec_macro * 100.0),
            "f1_score": float(f1_macro * 100.0),
            "hamming_loss_percent": float(hl * 100.0),
            "hamming_loss": float(hl),
            "jaccard_score": float(js * 100.0),
            "matthews_corrcoef": float(mcc * 100.0),
            "cohens_kappa": float(ck * 100.0),
        },
        "classwise": classwise_results,
        "confusion_matrix": cm.tolist()
    }


def format_metrics_table(metrics: Dict[str, Any]) -> pd.DataFrame:
    """Formats class-wise metrics into a clean DataFrame matching Table 6."""
    rows = []
    for cname, stats in metrics["classwise"].items():
        rows.append({
            "Tumor Types": cname.capitalize(),
            "Precision (%)": f"{stats['precision']:.2f}%",
            "Recall (%)": f"{stats['recall']:.2f}%",
            "F1-score (%)": f"{stats['f1_score']:.2f}%"
        })
    return pd.DataFrame(rows)


def print_paper_comparison(model_name: str, calculated_metrics: Dict[str, Any]):
    """
    Compares the calculated model metrics against the published TumorGANet benchmark.
    """
    sm = calculated_metrics["summary"]
    pb = config.PAPER_BENCHMARKS.get("TumorGANet", {})

    print(f"\n{'='*70}")
    print(f" Performance Comparison: {model_name} vs Published TumorGANet (Table 5)")
    print(f"{'='*70}")
    print(f"{'Metric':<25} | {'Achieved':<15} | {'Published Paper':<15}")
    print(f"{'-'*70}")
    print(f"{'Accuracy (%)':<25} | {sm['accuracy']:<15.2f} | {pb.get('accuracy', 99.53):<15.2f}")
    print(f"{'Precision (%)':<25} | {sm['precision']:<15.2f} | {pb.get('precision', 100.0):<15.2f}")
    print(f"{'Recall (%)':<25} | {sm['recall']:<15.2f} | {pb.get('recall', 100.0):<15.2f}")
    print(f"{'F1 Score (%)':<25} | {sm['f1_score']:<15.2f} | {pb.get('f1_score', 99.0):<15.2f}")
    print(f"{'Hamming Loss (%)':<25} | {sm['hamming_loss_percent']:<15.2f} | {pb.get('hamming_loss', 0.20):<15.2f}")
    print(f"{'Matthews Corr. Coeff. (%)':<25} | {sm['matthews_corrcoef']:<15.2f} | {pb.get('mcc', 99.50):<15.2f}")
    print(f"{'Jaccard Score (%)':<25} | {sm['jaccard_score']:<15.2f} | {pb.get('jaccard_score', 98.50):<15.2f}")
    ck_label = "Cohen's Kappa (%)"
    print(f"{ck_label:<25} | {sm['cohens_kappa']:<15.2f} | {pb.get('cohens_kappa', 99.50):<15.2f}")
    print(f"{'='*70}\n")

