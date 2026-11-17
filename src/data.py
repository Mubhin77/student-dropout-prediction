
import pandas as pd
from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split

from src.config import (
    FEATURES,
    TARGET_COLUMN,
    RANDOM_STATE,
    TEST_SIZE
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

def load_dataset(path=None):

    if path is None:
        path = PROJECT_ROOT / "data" / "raw" / "data.csv"

    df = pd.read_csv(path, sep=";")
    df.columns = df.columns.str.strip()

    df = df[df["Target"].isin(["Dropout", "Graduate"])].copy()
    df["target"] = (df["Target"] == "Dropout").astype(int)

    return df

def prepare_features(df):
    """Select only approved enrollment-time features."""

    missing_features = [
        col for col in FEATURES
        if col not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing required features: {missing_features}"
        )

    X = df[FEATURES].copy()
    y = df[TARGET_COLUMN].copy()

    return X, y


def split_data(X, y):
    """Create a stratified training and test split."""

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


def create_data_split():

    df = load_dataset()

    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = split_data(X, y)

    print("Dataset successfully prepared")
    print("-" * 40)

    print("Total records:", len(df))
    print("Features:", len(FEATURES))

    print("\nTraining records:", len(X_train))
    print("Testing records:", len(X_test))

    print("\nTraining target distribution:")
    print(y_train.value_counts(normalize=True).round(3))

    print("\nTesting target distribution:")
    print(y_test.value_counts(normalize=True).round(3))

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    create_data_split()
