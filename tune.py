from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict

from train import compute_class_weight

RANDOM_STATE = 42

# Small grid: keep trees shallow and avoid overfitting on this dataset.
PARAM_GRID = {
    "max_depth":     [3, 4, 5],
    "n_estimators":  [100, 200],
    "learning_rate": [0.05, 0.1],
    "subsample":     [0.7, 0.9],
}


def tune_hyperparameters(train_features, train_label):
    """Search for the best XGBoost hyperparameters using stratified k-fold CV.

    After finding the best parameter combination, runs cross_val_predict with
    the best estimator to produce out-of-fold probabilities on the training set.
    These are used for plotting CV performance without additional data leakage.

    Uses AUPRC as the scoring metric since the target is highly imbalanced.
    scale_pos_weight is fixed before the search so the grid focuses only on
    tree structure and learning parameters.

    Args:
        train_features: DataFrame of input features for the training set.
        train_label: Series of binary labels (0/1) for the training set.

    Returns:
        Tuple of:
            best_params (dict): Best hyperparameter values from the grid search.
            best_cv_auprc (float): Best mean CV AUPRC score across the grid.
            cv_train_probs (ndarray): Out-of-fold predicted fraud probabilities
                                     for the full training set using best params.
    """
    class_weight = compute_class_weight(train_label)

    base_model = XGBClassifier(
        scale_pos_weight=class_weight,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        eval_metric="aucpr",
        n_jobs=1,  # GridSearchCV parallelises across folds; avoid nested parallelism
    )

    cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    search = GridSearchCV(
        estimator=base_model,
        param_grid=PARAM_GRID,
        scoring="average_precision",
        cv=cv_splitter,
        n_jobs=-1,
        verbose=1,
    )

    search.fit(train_features, train_label)

    print(f"  Best CV AUPRC: {search.best_score_:.4f}")
    print(f"  Best params:   {search.best_params_}")

    # Get out-of-fold probabilities using the best estimator for plotting
    cv_train_probs = cross_val_predict(
        search.best_estimator_,
        train_features,
        train_label,
        cv=cv_splitter,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]

    return search.best_params_, search.best_score_, cv_train_probs
