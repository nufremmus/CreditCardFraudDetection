from data import load_data, split_out_of_time
from features import get_features
from tune import tune_hyperparameters
from train import train_final
from evaluate import evaluate, print_results, save_results
from plot import save_roc_pr_plot

CONFIGS = [
    {"include_time": True,  "label": "With Time"},
    {"include_time": False, "label": "Without Time"},
]


def main():
    """Run fraud detection training and evaluation for all feature configurations.

    Pipeline:
        1. Load and split data into train (80%) and out-of-time test (20%) by Time.
        2. For each config (with/without Time feature):
            a. Tune hyperparameters via grid search (CV is done internally).
            b. Train a final model on all training data with best params.
            c. Evaluate on the OOT test set and log all metrics.
            d. Save ROC and Precision-Recall plots grouped by config.
        3. Save all results to results_summary.csv.
    """
    print("Loading data...")
    transactions = load_data()
    train_transactions, test_transactions = split_out_of_time(transactions)

    print(
        f"Train: {len(train_transactions):,} rows  |  "
        f"fraud rate: {train_transactions['Class'].mean():.4%}"
    )
    print(
        f"Test:  {len(test_transactions):,} rows  |  "
        f"fraud rate: {test_transactions['Class'].mean():.4%}"
    )

    all_results = []

    for config in CONFIGS:
        config_label = config["label"]
        feature_cols = get_features(include_time=config["include_time"])

        train_features = train_transactions[feature_cols]
        train_label = train_transactions["Class"]
        test_features = test_transactions[feature_cols]
        test_label = test_transactions["Class"]

        print(f"\n{'#' * 50}")
        print(f"  CONFIG: {config_label}")
        print(f"{'#' * 50}")

        print("\n[Hyperparameter tuning]")
        best_params, best_cv_auprc, cv_train_probs = tune_hyperparameters(
            train_features, train_label
        )

        print("\n  Best hyperparameter combination:")
        for param_name, param_value in best_params.items():
            print(f"    {param_name}: {param_value}")
        print(f"  Best CV AUPRC from tuning: {best_cv_auprc:.4f}")

        print("\n[Final model — OOT test set]")
        final_model = train_final(train_features, train_label, params=best_params)
        test_fraud_probs = final_model.predict_proba(test_features)[:, 1]

        test_results = evaluate(
            test_label, test_fraud_probs, label=f"{config_label} — OOT Test"
        )
        print_results(test_results)
        all_results.append(test_results)

        print("\n[Saving plots]")
        save_roc_pr_plot(
            train_label=train_label,
            train_cv_probs=cv_train_probs,
            test_label=test_label,
            test_probs=test_fraud_probs,
            config_label=config_label,
        )

    save_results(all_results)


if __name__ == "__main__":
    main()
