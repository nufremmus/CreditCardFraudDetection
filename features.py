PCA_FEATURE_COLS = [f"V{i}" for i in range(1, 29)]
BASE_FEATURES = PCA_FEATURE_COLS + ["Amount"]


def get_features(include_time=True):
    """Return the list of input feature column names for model training.

    Args:
        include_time: If True, include the Time column as a feature.
                      Experimenting with this tests whether transaction
                      timing carries predictive signal for fraud.

    Returns:
        List of column name strings.
    """
    if include_time:
        return BASE_FEATURES + ["Time"]
    return BASE_FEATURES
