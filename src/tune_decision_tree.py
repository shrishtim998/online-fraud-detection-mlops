
from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
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

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

# 1. Load the existing train/test split
X_train = pd.read_parquet(DATA_DIR / "X_train.parquet")
X_test = pd.read_parquet(DATA_DIR / "X_test.parquet")

y_train = pd.read_parquet(
    DATA_DIR / "y_train.parquet"
).squeeze().astype(int)

y_test = pd.read_parquet(
    DATA_DIR / "y_test.parquet"
).squeeze().astype(int)

# 2. Take a reproducible 10% stratified sample of TRAINING data.
# This preserves the approximate original fraud prevalence.
train = X_train.copy()
train["_target"] = y_train.to_numpy()

sample = train.groupby(
    "_target", group_keys=False
).sample(frac=0.10, random_state=42)

y_sample = sample.pop("_target").astype(int)

# 3. Encode transaction type
X_sample = pd.get_dummies(
    sample, columns=["type"], dtype=int
)

X_test_encoded = pd.get_dummies(
    X_test, columns=["type"], dtype=int
)

# Ensure training and test columns match
X_test_encoded = X_test_encoded.reindex(
    columns=X_sample.columns,
    fill_value=0,
)

print("Training sample rows:", len(X_sample))
print("Fraud cases in sample:", int(y_sample.sum()))
print("Fraud prevalence in sample:", round(float(y_sample.mean()), 6))

# 4. Configure stratified cross-validation
cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42,
)

# 5. Hyperparameter search
parameter_grid = {
    "max_depth": [8, 12, 16, 20, None],
    "min_samples_leaf": [2, 5, 10, 20],
    "min_samples_split": [2, 10, 20],
    # balanced_subsample is NOT supported by DecisionTreeClassifier
    "class_weight": ["balanced", None],
}

search = RandomizedSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
    param_distributions=parameter_grid,
    n_iter=8,
    scoring="average_precision",
    cv=cv,
    random_state=42,
    n_jobs=2,
    verbose=1,
    refit=True,
    error_score="raise",
)

print("\nStarting Decision Tree tuning...")
print("Using 3-fold stratified cross-validation.")

search.fit(X_sample, y_sample)

best_model = search.best_estimator_

# 6. Evaluate the selected model on the held-out test set
# The test set is not used during hyperparameter selection.
probabilities = best_model.predict_proba(
    X_test_encoded
)[:, 1]

threshold = 0.5
predictions = (probabilities >= threshold).astype(int)

metrics = {
    "model": "Tuned Decision Tree",
    "best_params": search.best_params_,
    "best_cv_pr_auc": float(search.best_score_),
    "test_accuracy": float(
        (predictions == y_test.to_numpy()).mean()
    ),
    "test_precision": float(
        precision_score(y_test, predictions, zero_division=0)
    ),
    "test_recall": float(
        recall_score(y_test, predictions, zero_division=0)
    ),
    "test_f1": float(
        f1_score(y_test, predictions, zero_division=0)
    ),
    "test_roc_auc": float(
        roc_auc_score(y_test, probabilities)
    ),
    "test_pr_auc": float(
        average_precision_score(y_test, probabilities)
    ),
    "test_confusion_matrix": confusion_matrix(
        y_test, predictions
    ).tolist(),
    "threshold": threshold,
    "training_sample_size": int(len(X_sample)),
    "cv_folds": 3,
    "random_search_iterations": 8,
}

# 7. Save the tuned model and its results
joblib.dump(
    best_model,
    MODEL_DIR / "decision_tree_tuned.pkl",
)

with open(
    REPORT_DIR / "decision_tree_tuning_metrics.json", "w"
) as f:
    json.dump(metrics, f, indent=4)

print("\nBEST PARAMETERS")
print(search.best_params_)
print(f"Best cross-validation PR-AUC: {search.best_score_:.4f}")

print("\nHELD-OUT TEST RESULTS")
for key in [
    "test_accuracy",
    "test_precision",
    "test_recall",
    "test_f1",
    "test_roc_auc",
    "test_pr_auc",
]:
    print(f"{key}: {metrics[key]:.4f}")

print("\nConfusion Matrix:")
print(metrics["test_confusion_matrix"])

print("\nSaved model: models/decision_tree_tuned.pkl")
print("Saved metrics: reports/decision_tree_tuning_metrics.json")
