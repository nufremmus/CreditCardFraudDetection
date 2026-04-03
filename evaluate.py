import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)

# Decision thresholds to evaluate: 0.1%, 0.2%, 0.3%, 0.5% predicted fraud probability
DECISION_THRESHOLDS = [0.001, 0.002, 0.003, 0.005]


def evaluate(true_labels, predicted_fraud_probs, label=""):
    """Compute classification metrics across multiple decision thresholds.

    AUPRC is the primary metric for imbalanced fraud detection because it
    focuses on the minority class and is not inflated by true negatives.

    Args:
        true_labels: Array-like of ground truth binary labels (0/1).
        predicted_fraud_probs: Array-like of predicted fraud probabilities.
        label: Human-readable name for this result set (used in output).

    Returns:
        Dict with AUPRC, ROC-AUC, and per-threshold Precision/Recall/F1.
    """
    results = {
        "label": label,
        "AUPRC": average_precision_score(true_labels, predicted_fraud_probs),
        "ROC-AUC": roc_auc_score(true_labels, predicted_fraud_probs),
    }

    for threshold in DECISION_THRESHOLDS:
        predicted_labels = (predicted_fraud_probs >= threshold).astype(int)
        results[f"Precision@{threshold}"] = precision_score(
            true_labels, predicted_labels, zero_division=0
        )
        results[f"Recall@{threshold}"] = recall_score(
            true_labels, predicted_labels, zero_division=0
        )
        results[f"F1@{threshold}"] = f1_score(
            true_labels, predicted_labels, zero_division=0
        )

    return results


def print_results(results):
    """Print a formatted summary of evaluation metrics to stdout.

    Args:
        results: Dict returned by evaluate().
    """
    print(f"\n{'=' * 50}")
    print(f"  {results['label']}")
    print(f"{'=' * 50}")
    print(f"  AUPRC:   {results['AUPRC']:.4f}")
    print(f"  ROC-AUC: {results['ROC-AUC']:.4f}")
    print()
    for threshold in DECISION_THRESHOLDS:
        precision = results[f"Precision@{threshold}"]
        recall = results[f"Recall@{threshold}"]
        f1 = results[f"F1@{threshold}"]
        print(
            f"  Threshold {threshold * 100:.1f}%:  "
            f"Precision={precision:.3f}  Recall={recall:.3f}  F1={f1:.3f}"
        )


def save_results(all_results, path="results_summary.csv"):
    """Save all configuration results to a CSV file.

    Args:
        all_results: List of result dicts from evaluate().
        path: Output file path.
    """
    pd.DataFrame(all_results).to_csv(path, index=False)
    print(f"\nResults saved to {path}")
