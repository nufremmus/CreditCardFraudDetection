from data import load_data, split_out_of_time
from features import get_features
from tune import tune_hyperparameters
from train import cross_validate, train_final
from evaluate import evaluate, print_results, save_results

CONFIGS = [
    {"include_time": True,  "label": "With Time"},
    {"include_time": False, "label": "Without Time"},
]


def main():
    """Run fraud detection training and evaluation for all feature configurations.

    Pipeline:
        1. Load and split data into train (80%) and out-of-time test (20%) by Time.
        2. For each config (with/without Time feature):
            a. Tune hyperparameters via grid search on the training set.
            b. Run stratified k-fold CV with best params; report OOF metrics.
            c. Train a final model on all training data; evaluate on OOT test set.
        3. Save all results to results_summary.csv.
    """
    print("Loading data...")
    transactions = load_data()
    train_transactions, test_transactions = split_out_of_time(transactions)

    print(f"Train: {len(train_transactions):,} rows  |  fraud rate: {train_transactions['Class'].mean():.4%}")
    print(f"Test:  {len(test_transactions):,} rows  |  fraud rate: {test_transactions['Class'].mean():.4%}")

    all_results = []

    for config in CONFIGS:
        config_label = config["label"]
        feature_cols = get_features(include_time=config["include_time"])

        train_features = train_transactions[feature_cols]
        train_labels = train_transactions["Class"]
        test_features = test_transactions[feature_cols]
        test_labels = test_transactions["Class"]

        print(f"\n{'#' * 50}")
        print(f"  CONFIG: {config_label}")
        print(f"{'#' * 50}")

        print("\n[Hyperparameter tuning]")
        best_params = tune_hyperparameters(train_features, train_labels)

        print("\n[Cross-validation with best params]")
        oof_fraud_probs = cross_validate(train_features, train_labels, params=best_params)
        cv_results = evaluate(
            train_labels, oof_fraud_probs, label=f"{config_label} — CV (OOF)"
        )
        print_results(cv_results)
        all_results.append(cv_results)

        print("\n[Final model — OOT test set]")
        final_model = train_final(train_features, train_labels, params=best_params)
        test_fraud_probs = final_model.predict_proba(test_features)[:, 1]
        test_results = evaluate(
            test_labels, test_fraud_probs, label=f"{config_label} — OOT Test"
        )
        print_results(test_results)
        all_results.append(test_results)

    save_results(all_results)


if __name__ == "__main__":
    main()
