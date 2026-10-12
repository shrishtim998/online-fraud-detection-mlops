
from pathlib import Path
import json
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

TARGET = "isFraud"
RANDOM_STATE = 42
SAMPLE_SIZE = 100_000


def main():
    start = time.time()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading saved training and test data...")

    X_train = pd.read_parquet(DATA_DIR / "X_train.parquet")
    y_train = pd.read_parquet(DATA_DIR / "y_train.parquet")[TARGET]

    X_test = pd.read_parquet(DATA_DIR / "X_test.parquet")
    y_test = pd.read_parquet(DATA_DIR / "y_test.parquet")[TARGET]

    print(f"Training rows available: {len(X_train):,}")
    print(f"Test rows available: {len(X_test):,}")

    # Stratified sample from training data only
    sample_idx, _ = train_test_split(
        range(len(y_train)),
        train_size=SAMPLE_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )

    X_fit = X_train.iloc[sample_idx].copy()
    y_fit = y_train.iloc[sample_idx].copy()

    del X_train, y_train, sample_idx

    print(f"\nRandom Forest training sample: {len(X_fit):,}")
    print("Training class distribution:")
    print(y_fit.value_counts())

    # Random Forest supports categorical data only after encoding.
    # This dataset has a small number of transaction types.
    X_fit = pd.get_dummies(X_fit, columns=["type"], dtype="int8")
    X_test = pd.get_dummies(X_test, columns=["type"], dtype="int8")
    X_fit, X_test = X_fit.align(
        X_test, join="left", axis=1, fill_value=0
    )

    print("\nTraining Random Forest...")

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=2,
    )

    model.fit(X_fit, y_fit)

    del X_fit, y_fit

    print("Training completed.")
    print("\nEvaluating on the full test set...")

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": "Random Forest",
        "training_sample_size": SAMPLE_SIZE,
        "test_size": int(len(y_test)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(
            precision_score(y_test, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, y_pred, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_test, y_pred, zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "pr_auc_average_precision": float(
            average_precision_score(y_test, y_prob)
        ),
        "threshold": 0.5,
    }

    print("\n--- Random Forest Metrics ---")
    for name, value in metrics.items():
        print(f"{name}: {value}")

    metrics_path = REPORT_DIR / "random_forest_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    cm = confusion_matrix(y_test, y_pred)
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Legitimate", "Fraud"],
    )
    display.plot(values_format="d")
    plt.title("Random Forest - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(
        REPORT_DIR / "random_forest_confusion_matrix.png",
        dpi=150,
    )
    plt.close()

    joblib.dump(
        model,
        MODEL_DIR / "random_forest.pkl",
        compress=3,
    )

    print(f"\nMetrics saved: {metrics_path}")
    print(
        "Confusion matrix saved: "
        f"{REPORT_DIR / 'random_forest_confusion_matrix.png'}"
    )
    print(f"Model saved: {MODEL_DIR / 'random_forest.pkl'}")
    print(f"Elapsed time: {time.time() - start:.1f} seconds")


if __name__ == "__main__":
    main()
