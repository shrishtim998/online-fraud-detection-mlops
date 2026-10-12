
from pathlib import Path
import json

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"

REPORT_DIR.mkdir(exist_ok=True)

# Load the existing train/test split
X_train = pd.read_parquet(DATA_DIR / "X_train.parquet")
X_test = pd.read_parquet(DATA_DIR / "X_test.parquet")
y_train = pd.read_parquet(DATA_DIR / "y_train.parquet").squeeze()
y_test = pd.read_parquet(DATA_DIR / "y_test.parquet").squeeze()

# Use a manageable, reproducible sample from the training set
train = X_train.copy()
train["_target"] = y_train.to_numpy()

fraud = train[train["_target"] == 1]
legitimate = train[train["_target"] == 0]

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

# Encode categorical transaction type
X_sample = pd.get_dummies(X_sample, columns=["type"], dtype=int)
X_test_encoded = pd.get_dummies(X_test, columns=["type"], dtype=int)
X_test_encoded = X_test_encoded.reindex(
    columns=X_sample.columns, fill_value=0
)

# Use a smaller training subset for SMOTE to limit memory usage
# Preserve all fraud cases and sample up to 20,000 legitimate cases
sample_for_smote = pd.concat([
    X_sample[y_sample == 0].sample(
        n=min(20_000, int((y_sample == 0).sum())),
        random_state=42,
    ),
    X_sample[y_sample == 1],
])

y_smote = y_sample.loc[sample_for_smote.index]
X_smote = sample_for_smote

results = []

def evaluate_model(name, X_fit, y_fit, class_weight=None):
    model = RandomForestClassifier(
        n_estimators=60,
        max_depth=12,
        min_samples_leaf=5,
        class_weight=class_weight,
        random_state=42,
        n_jobs=2,
    )
    model.fit(X_fit, y_fit)

    probabilities = model.predict_proba(X_test_encoded)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    results.append({
        "method": name,
        "precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "pr_auc": average_precision_score(y_test, probabilities),
        "training_rows": len(X_fit),
    })

# Approach 1: Class weighting
evaluate_model(
    "Class weighting",
    X_sample,
    y_sample,
    class_weight="balanced",
)

# Approach 2: Random undersampling
undersampled = pd.concat([
    X_sample[y_sample == 0].sample(
        n=min(10_000, int((y_sample == 0).sum())),
        random_state=42,
    ),
    X_sample[y_sample == 1],
])
y_under = y_sample.loc[undersampled.index]

evaluate_model(
    "Random undersampling",
    undersampled,
    y_under,
)

# Approach 3: SMOTE on numeric features and one-hot encoded categories
smote = SMOTE(random_state=42, k_neighbors=3)
X_resampled, y_resampled = smote.fit_resample(X_smote, y_smote)

evaluate_model(
    "SMOTE oversampling",
    X_resampled,
    y_resampled,
)

comparison = pd.DataFrame(results)
comparison.to_csv(
    REPORT_DIR / "imbalance_comparison.csv",
    index=False,
)

print("\nCLASS-IMBALANCE STRATEGY COMPARISON")
print(comparison.to_string(
    index=False,
    formatters={
        col: lambda x: f"{x:.4f}"
        for col in ["precision", "recall", "f1", "roc_auc", "pr_auc"]
    },
))
print("\nSaved: reports/imbalance_comparison.csv")
