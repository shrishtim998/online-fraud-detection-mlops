
from pathlib import Path
import json
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

# --------------------------------------------------
# 1. Paths and configuration
# --------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

TARGET = "isFraud"
RANDOM_STATE = 42
SAMPLE_SIZE = 300_000

X_TRAIN_PATH = DATA_DIR / "X_train.parquet"
X_TEST_PATH = DATA_DIR / "X_test.parquet"
Y_TRAIN_PATH = DATA_DIR / "y_train.parquet"
Y_TEST_PATH = DATA_DIR / "y_test.parquet"


def main():
    start = time.time()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # 2. Load existing train/test splits
    # --------------------------------------------------
    print("Loading saved datasets...")

    for path in [
        X_TRAIN_PATH, X_TEST_PATH,
        Y_TRAIN_PATH, Y_TEST_PATH
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing file: {path}\n"
                "Run src/preprocessing.py first."
            )

    X_train = pd.read_parquet(X_TRAIN_PATH)
    X_test = pd.read_parquet(X_TEST_PATH)

    y_train = pd.read_parquet(Y_TRAIN_PATH)[TARGET]
    y_test = pd.read_parquet(Y_TEST_PATH)[TARGET]

    print(f"Training rows available: {len(X_train):,}")
    print(f"Testing rows available: {len(X_test):,}")

    # --------------------------------------------------
    # 3. Stratified sample from training data only
    # --------------------------------------------------
    if SAMPLE_SIZE > len(X_train):
        raise ValueError(
            f"SAMPLE_SIZE={SAMPLE_SIZE:,} exceeds "
            f"training rows={len(X_train):,}"
        )

    sample_indices, _ = train_test_split(
        range(len(y_train)),
        train_size=SAMPLE_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )

    X_fit = X_train.iloc[sample_indices].copy()
    y_fit = y_train.iloc[sample_indices].copy()

    print(f"\nSample used for fitting: {len(X_fit):,}")
    print("Sample class distribution:")
    print(y_fit.value_counts())

    # Release the full training feature matrix before fitting.
    del X_train, y_train, sample_indices

    # --------------------------------------------------
    # 4. Define preprocessing
    # --------------------------------------------------
    numeric_features = [
        "step",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
    ]

    categorical_features = ["type"]

    missing_columns = [
        col for col in numeric_features + categorical_features
        if col not in X_fit.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Missing expected features: {missing_columns}"
        )

    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                numeric_features,
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    # --------------------------------------------------
    # 5. Train baseline Logistic Regression
    # --------------------------------------------------
    model = Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=200,
                    solver="lbfgs",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    print("\nTraining Logistic Regression...")
    model.fit(X_fit, y_fit)
    print("Training completed.")

    # --------------------------------------------------
    # 6. Evaluate on the complete untouched test set
    # --------------------------------------------------
    print("\nPredicting on the full test set...")

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": "Logistic Regression",
        "training_sample_size": int(len(X_fit)),
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

    print("\n--- Baseline Evaluation ---")
    for name, value in metrics.items():
        print(f"{name}: {value}")

    print("\n--- Classification Report ---")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["Legitimate", "Fraud"],
            zero_division=0,
        )
    )

    # --------------------------------------------------
    # 7. Save metrics and confusion matrix
    # --------------------------------------------------
    metrics_path = REPORT_DIR / "baseline_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    cm = confusion_matrix(y_test, y_pred)
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Legitimate", "Fraud"],
    )
    display.plot(values_format="d")
    plt.title("Logistic Regression - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(
        REPORT_DIR / "baseline_confusion_matrix.png",
        dpi=150,
    )
    plt.close()

    # --------------------------------------------------
    # 8. Save trained pipeline
    # --------------------------------------------------
    model_path = MODEL_DIR / "logistic_regression.pkl"
    joblib.dump(model, model_path, compress=3)

    print(f"\nModel saved: {model_path}")
    print(f"Metrics saved: {metrics_path}")
    print(
        "Confusion matrix saved: "
        f"{REPORT_DIR / 'baseline_confusion_matrix.png'}"
    )
    print(f"Total elapsed time: {time.time() - start:.1f} seconds")


if __name__ == "__main__":
    main()
