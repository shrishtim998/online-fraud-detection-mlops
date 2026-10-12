
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load the existing stratified train/test split
X_train = pd.read_parquet(DATA_DIR / "X_train.parquet")
X_test = pd.read_parquet(DATA_DIR / "X_test.parquet")

y_train = pd.read_parquet(
    DATA_DIR / "y_train.parquet"
).squeeze().astype(int)

y_test = pd.read_parquet(
    DATA_DIR / "y_test.parquet"
).squeeze().astype(int)

# 2. Keep all fraud cases and sample legitimate transactions
train_data = X_train.copy()
train_data["_target"] = y_train.to_numpy()

fraud_rows = train_data[train_data["_target"] == 1]
legitimate_rows = train_data[train_data["_target"] == 0]

legitimate_sample = legitimate_rows.sample(
    n=min(100_000, len(legitimate_rows)),
    random_state=42,
)

training_data = pd.concat(
    [legitimate_sample, fraud_rows],
    ignore_index=True,
).sample(frac=1, random_state=42)

y_sample = training_data.pop("_target").astype(int)
X_sample = training_data

# 3. Define preprocessing inside the pipeline
categorical_features = ["type"]


numeric_features = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "origin_balance_discrepancy",
    "destination_balance_discrepancy",
]


preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features,
        ),
        ("numeric", "passthrough", numeric_features),
    ]
)

# 4. Combine preprocessing and Random Forest
final_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestClassifier(
                n_estimators=100,
                max_depth=12,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=42,
                n_jobs=2,
            ),
        ),
    ]
)

# 5. Train the pipeline
print("Training final preprocessing + Random Forest pipeline...")
final_pipeline.fit(X_sample, y_sample)

# 6. Evaluate on the held-out test set
print("Evaluating on held-out test data...")

probabilities = final_pipeline.predict_proba(X_test)[:, 1]
threshold = 0.5
predictions = (probabilities >= threshold).astype(int)

metrics = {
    "model": "Random Forest Pipeline",
    "threshold": threshold,
    "training_sample_size": int(len(X_sample)),
    "test_size": int(len(X_test)),
    "accuracy": float(accuracy_score(y_test, predictions)),
    "precision": float(
        precision_score(y_test, predictions, zero_division=0)
    ),
    "recall": float(
        recall_score(y_test, predictions, zero_division=0)
    ),
    "f1_score": float(
        f1_score(y_test, predictions, zero_division=0)
    ),
    "roc_auc": float(roc_auc_score(y_test, probabilities)),
    "pr_auc": float(
        average_precision_score(y_test, probabilities)
    ),
    "confusion_matrix": confusion_matrix(
        y_test, predictions
    ).tolist(),
}

# 7. Save the complete fitted pipeline
pipeline_path = MODEL_DIR / "final_fraud_pipeline.pkl"
joblib.dump(final_pipeline, pipeline_path)
print(f"Saved pipeline: {pipeline_path}")

# 8. Reload the saved pipeline
reloaded_pipeline = joblib.load(pipeline_path)
reloaded_probabilities = (
    reloaded_pipeline.predict_proba(X_test)[:, 1]
)

# Compare floating-point predictions within a small tolerance
max_difference = float(
    np.max(np.abs(probabilities - reloaded_probabilities))
)

predictions_match = bool(
    np.allclose(
        probabilities,
        reloaded_probabilities,
        rtol=1e-7,
        atol=1e-9,
    )
)

metrics["reload_predictions_match"] = predictions_match
metrics["max_reload_probability_difference"] = max_difference

# Save metrics even if validation fails
metrics_path = REPORT_DIR / "final_pipeline_metrics.json"

with open(metrics_path, "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=4)

print(f"Saved metrics: {metrics_path}")

print("\nFINAL PIPELINE RESULTS")
for key, value in metrics.items():
    print(f"{key}: {value}")

if not predictions_match:
    raise AssertionError(
        "Reloaded pipeline predictions differ beyond the allowed tolerance."
    )

print("\nReload validation: PASSED")
