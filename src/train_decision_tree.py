
from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from sklearn.utils import resample

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

# Load the existing stratified train/test split
X_train = pd.read_parquet(DATA_DIR / "X_train.parquet")
X_test = pd.read_parquet(DATA_DIR / "X_test.parquet")
y_train = pd.read_parquet(DATA_DIR / "y_train.parquet").squeeze()
y_test = pd.read_parquet(DATA_DIR / "y_test.parquet").squeeze()

# Use a manageable stratified sample for an 8 GB RAM computer
train_data = X_train.copy()
train_data["_target"] = y_train.to_numpy()

fraud = train_data[train_data["_target"] == 1]
legitimate = train_data[train_data["_target"] == 0]

legitimate_sample = resample(
    legitimate,
    replace=False,
    n_samples=min(100_000, len(legitimate)),
    random_state=42,
)

sample = pd.concat([legitimate_sample, fraud], ignore_index=True)
sample = sample.sample(frac=1, random_state=42)

y_sample = sample.pop("_target").astype(int)
X_sample = sample

# Encode transaction type consistently with the model features
X_sample = pd.get_dummies(X_sample, columns=["type"], dtype=int)
X_test_encoded = pd.get_dummies(X_test, columns=["type"], dtype=int)

X_test_encoded = X_test_encoded.reindex(
    columns=X_sample.columns,
    fill_value=0,
)

# Train Decision Tree with class weighting for class imbalance
model = DecisionTreeClassifier(
    max_depth=12,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42,
)

model.fit(X_sample, y_sample)

# Evaluate using the existing held-out test set
probabilities = model.predict_proba(X_test_encoded)[:, 1]
predictions = (probabilities >= 0.5).astype(int)

metrics = {
    "model": "Decision Tree",
    "accuracy": float(accuracy_score(y_test, predictions)),
    "precision": float(precision_score(y_test, predictions, zero_division=0)),
    "recall": float(recall_score(y_test, predictions, zero_division=0)),
    "f1": float(f1_score(y_test, predictions, zero_division=0)),
    "roc_auc": float(roc_auc_score(y_test, probabilities)),
    "average_precision": float(average_precision_score(y_test, probabilities)),
    "threshold": 0.5,
    "training_sample_size": int(len(X_sample)),
}

joblib.dump(model, MODEL_DIR / "decision_tree.pkl")

with open(REPORT_DIR / "decision_tree_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("\nDecision Tree Evaluation")
for metric, value in metrics.items():
    print(f"{metric}: {value}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("\nSaved model: models/decision_tree.pkl")
print("Saved metrics: reports/decision_tree_metrics.json")
