import pandas as pd

DATA_PATH = "creditcard.csv"
OOT_FRACTION = 0.2  # last 20% by Time reserved as out-of-time test set


def load_data(path=DATA_PATH):
    """Load the credit card transaction CSV into a DataFrame.

    Args:
        path: Path to the CSV file.

    Returns:
        DataFrame with all 31 columns (V1-V28, Time, Amount, Class).
    """
    return pd.read_csv(path)


def split_out_of_time(transactions, oot_fraction=OOT_FRACTION):
    """Split transactions into train and test by Time.

    The test set contains the most recent transactions (highest Time values),
    ensuring no future data leaks into training — an out-of-time split.

    Args:
        transactions: Full transactions DataFrame.
        oot_fraction: Fraction of the time range to reserve for testing.

    Returns:
        (train_transactions, test_transactions) as separate DataFrames.
    """
    time_cutoff = transactions["Time"].quantile(1 - oot_fraction)
    train_transactions = transactions[transactions["Time"] < time_cutoff].reset_index(drop=True)
    test_transactions = transactions[transactions["Time"] >= time_cutoff].reset_index(drop=True)
    return train_transactions, test_transactions
