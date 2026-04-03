import logging
import os

import matplotlib.pyplot as plt
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)

logger = logging.getLogger(__name__)

PLOTS_DIR = "plots"


def save_roc_pr_plot(
    train_label,
    train_cv_probs,
    test_label,
    test_probs,
    config_label,
    output_dir=PLOTS_DIR,
):
    """Generate and save ROC and Precision-Recall curves for one configuration.

    Plots two curves per chart — CV performance on the training set and
    final model performance on the out-of-time test set — so both can be
    compared in a single figure.

    Args:
        train_label: True binary labels for the training set.
        train_cv_probs: Cross-validated fraud probabilities for the training set.
        test_label: True binary labels for the OOT test set.
        test_probs: Predicted fraud probabilities from the final model on the test set.
        config_label: Human-readable config name used in the title and filename.
        output_dir: Directory where the PNG will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(config_label, fontsize=14, fontweight="bold")

    curves = [
        (train_label, train_cv_probs, "CV (train)"),
        (test_label, test_probs, "OOT Test"),
    ]

    # ROC curves
    for true_labels, probs, curve_label in curves:
        fpr, tpr, _ = roc_curve(true_labels, probs)
        roc_auc = auc(fpr, tpr)
        ax_roc.plot(fpr, tpr, label=f"{curve_label}  (AUC = {roc_auc:.4f})")

    ax_roc.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curve")
    ax_roc.legend()

    # Precision-Recall curves
    for true_labels, probs, curve_label in curves:
        precision, recall, _ = precision_recall_curve(true_labels, probs)
        ap = average_precision_score(true_labels, probs)
        ax_pr.plot(recall, precision, label=f"{curve_label}  (AUPRC = {ap:.4f})")

    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-Recall Curve")
    ax_pr.legend()

    plt.tight_layout()

    filename = config_label.lower().replace(" ", "_") + ".png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Plot saved to %s", filepath)
