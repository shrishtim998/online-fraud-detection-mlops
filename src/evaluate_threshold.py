
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    average_precision_score,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

MODEL_PATH = MODEL_DIR / "random_forest.pkl"
TARGET = "isFraud"

# Keep memory use manageable on an 8 GB RAM PC.
EVAL_SAMPLE_SIZE = 200_000
RANDOM_STATE = 42

THRESHOLDS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading saved Random Forest model...")
    model = joblib.load(MODEL_PATH)

    print("Loading test data...")
    X_test = pd.read_parquet(DATA_DIR / "X_test.parquet")
    y_test = pd.read_parquet(DATA_DIR / "y_test.parquet")[TARGET]

    # Evaluate a reproducible sample to limit memory and processing time.
    if len(y_test) > EVAL_SAMPLE_SIZE:
        rng = np.random.default_rng(RANDOM_STATE)
        indices = rng.choice(
            len(y_test), size=EVAL_SAMPLE_SIZE, replace=False
        )
        X_test = X_test.iloc[indices].copy()
        y_test = y_test.iloc[indices].copy()

    print(f"Evaluation rows: {len(y_test):,}")

    # Match the one-hot encoding used during training.
    X_test = pd.get_dummies(
        X_test, columns=["type"], dtype="int8"
    )

    # Use the model's saved feature order.
    X_test = X_test.reindex(
        columns=model.feature_names_in_, fill_value=0
    )

    print("Calculating fraud probabilities...")
    probabilities = model.predict_proba(X_test)[:, 1]

    results = []

    for threshold in THRESHOLDS:
        predictions = (probabilities >= threshold).astype("int8")

        tn, fp, fn, tp = confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ).ravel()

        results.append({
            "threshold": threshold,
            "precision": float(
                precision_score(
                    y_test, predictions, zero_division=0
                )
            ),
            "recall": float(
                recall_score(
                    y_test, predictions, zero_division=0
                )
            ),
            "f1_score": float(
                f1_score(
                    y_test, predictions, zero_division=0
                )
            ),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_negatives": int(tn),
        })

    results_df = pd.DataFrame(results)
    csv_path = REPORT_DIR / "threshold_comparison.csv"
    results_df.to_csv(csv_path, index=False)

    best = max(results, key=lambda row: row["f1_score"])

    json_path = REPORT_DIR / "best_threshold.json"
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(best, file, indent=4)

    print("\n--- Threshold Comparison ---")
    print(results_df.round(4).to_string(index=False))

    print("\nBest threshold by F1-score:")
    print(json.dumps(best, indent=4))

    print(f"\nComparison saved: {csv_path}")
    print(f"Best threshold saved: {json_path}")


if __name__ == "__main__":
    main()
