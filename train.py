import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score

N_SPLITS = 5
RANDOM_STATE = 42

# Default hyperparameters used when no tuning results are provided.
DEFAULT_PARAMS = {
    "max_depth":     5,
    "n_estimators":  300,
    "learning_rate": 0.05,
    "subsample":     0.8,
}


def compute_class_weight(labels):
    """Compute scale_pos_weight as ratio of negative to positive samples.

    XGBoost uses this to upweight the minority (fraud) class during training,
    compensating for the severe class imbalance (~0.17% fraud rate).

    Args:
        labels: Series of binary labels (0 = genuine, 1 = fraud).

    Returns:
        Float ratio: count(genuine) / count(fraud).
    """
    return (labels == 0).sum() / (labels == 1).sum()


def make_model(class_weight, params=None, early_stopping_rounds=20):
    """Instantiate an XGBClassifier with the given hyperparameters.

    Args:
        class_weight: Value for scale_pos_weight (negative/positive ratio).
        params: Dict of hyperparameters. Falls back to DEFAULT_PARAMS if None.
        early_stopping_rounds: Stops early when eval metric stops improving.
                               Pass None to disable (e.g. for final training).

    Returns:
        Configured XGBClassifier (not yet fitted).
    """
    resolved_params = params if params is not None else DEFAULT_PARAMS
    return XGBClassifier(
        scale_pos_weight=class_weight,
        max_depth=resolved_params["max_depth"],
        n_estimators=resolved_params["n_estimators"],
        learning_rate=resolved_params["learning_rate"],
        subsample=resolved_params["subsample"],
        colsample_bytree=0.8,
        eval_metric="aucpr",
        early_stopping_rounds=early_stopping_rounds,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def cross_validate(train_features, train_label, params=None):
    """Run stratified k-fold cross-validation and return out-of-fold probabilities.

    Stratified splits preserve the fraud/genuine ratio in each fold.
    Out-of-fold (OOF) probabilities cover the full training set without
    leakage, giving an unbiased estimate of model performance.

    Args:
        train_features: DataFrame of input features for the training set.
        train_label: Series of binary labels (0/1) for the training set.
        params: Hyperparameter dict (e.g. from tune_hyperparameters).
                Falls back to DEFAULT_PARAMS if None.

    Returns:
        NumPy array of predicted fraud probabilities for each training row.
    """
    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof_fraud_probabilities = np.zeros(len(train_label))

    for fold_num, (fold_train_idx, fold_val_idx) in enumerate(
        splitter.split(train_features, train_label)
    ):
        fold_train_features = train_features.iloc[fold_train_idx]
        fold_val_features = train_features.iloc[fold_val_idx]
        fold_train_label = train_label.iloc[fold_train_idx]
        fold_val_label = train_label.iloc[fold_val_idx]

        model = make_model(compute_class_weight(fold_train_label), params=params)
        model.fit(
            fold_train_features,
            fold_train_label,
            eval_set=[(fold_val_features, fold_val_labels)],
            verbose=False,
        )
        oof_fraud_probabilities[fold_val_idx] = model.predict_proba(fold_val_features)[:, 1]

        fold_auprc = average_precision_score(
            fold_val_label, oof_fraud_probabilities[fold_val_idx]
        )
        print(
            f"  Fold {fold_num + 1}/{N_SPLITS}: "
            f"AUPRC = {fold_auprc:.4f}  "
            f"(best iteration: {model.best_iteration})"
        )

    return oof_fraud_probabilities


def train_final(train_features, train_label, params=None):
    """Train a final XGBoost model on the full training set.

    Early stopping is disabled since there is no held-out validation set
    at this stage. Uses a fixed iteration count from params instead.

    Args:
        train_features: DataFrame of input features for the full training set.
        train_label: Series of binary labels for the full training set.
        params: Hyperparameter dict (e.g. from tune_hyperparameters).
                Falls back to DEFAULT_PARAMS if None.

    Returns:
        Fitted XGBClassifier.
    """
    model = make_model(
        compute_class_weight(train_label),
        params=params,
        early_stopping_rounds=None,
    )
    model.fit(train_features, train_label)
    return model
