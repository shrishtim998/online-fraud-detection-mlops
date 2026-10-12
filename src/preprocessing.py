
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "onlinefraud.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

TARGET = "isFraud"
RANDOM_STATE = 42


def load_data():
    """Load and validate the original dataset."""
    df = pd.read_csv(DATA_PATH)

    if TARGET not in df.columns:
        raise ValueError(
            f"Target column '{TARGET}' not found. "
            f"Available columns: {df.columns.tolist()}"
        )

    if df[TARGET].isna().any():
        raise ValueError("Target column contains missing values.")

    if not set(df[TARGET].unique()).issubset({0, 1}):
        raise ValueError("Target must contain only 0 and 1.")

    print(f"Loaded dataset: {df.shape[0]:,} rows, {df.shape[1]} columns")
    return df



def prepare_data(df):
    """Create balance features and prepare model inputs."""

    # Separate target
    y = df[TARGET].astype("int8")

    # Feature Engineering: Balance Discrepancy
    df = df.copy()

    df["origin_balance_discrepancy"] = (
        df["oldbalanceOrg"]
        - df["amount"]
        - df["newbalanceOrig"]
    )

    df["destination_balance_discrepancy"] = (
        df["oldbalanceDest"]
        + df["amount"]
        - df["newbalanceDest"]
    )

    print("\nBalance discrepancy feature examples:")
    print(
        df[
            [
                "origin_balance_discrepancy",
                "destination_balance_discrepancy",
            ]
        ].head()
    )

    # Remove target, existing rule-based flag, and identifiers
    columns_to_drop = [
        "isFraud",
        "isFlaggedFraud",
        "nameOrig",
        "nameDest",
    ]

    X = df.drop(
        columns=[c for c in columns_to_drop if c in df.columns]
    ).copy()

    # Validate categorical feature
    if "type" in X.columns:
        X["type"] = X["type"].astype("category")

    # Validate missing values
    if X.isna().any().any():
        missing = X.isna().sum()
        raise ValueError(
            "Features contain missing values:\n"
            f"{missing[missing > 0]}"
        )

    print("\nFeatures used:")
    print(X.columns.tolist())
    print(f"Feature matrix shape: {X.shape}")

    return X, y



def split_data(X, y):
    """Create stratified training and test sets."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("\n--- Split Results ---")
    print(f"Training rows: {len(X_train):,}")
    print(f"Testing rows: {len(X_test):,}")

    print("\nTraining fraud distribution:")
    print(y_train.value_counts())
    print("\nTesting fraud distribution:")
    print(y_test.value_counts())

    return X_train, X_test, y_train, y_test


def main():
    df = load_data()
    X, y = prepare_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save splits for repeatable experiments.
    # Keep these files out of GitHub if they are large or sensitive.
    X_train.to_parquet(OUTPUT_DIR / "X_train.parquet", index=False)
    X_test.to_parquet(OUTPUT_DIR / "X_test.parquet", index=False)
    y_train.to_frame(TARGET).to_parquet(
        OUTPUT_DIR / "y_train.parquet", index=False
    )
    y_test.to_frame(TARGET).to_parquet(
        OUTPUT_DIR / "y_test.parquet", index=False
    )

    print(f"\nProcessed files saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

