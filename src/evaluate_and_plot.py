
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"

FIG_DIR.mkdir(parents=True, exist_ok=True)

X_test = pd.read_parquet(DATA_DIR / "X_test.parquet")
y_test = pd.read_parquet(
    DATA_DIR / "y_test.parquet"
).squeeze().astype(int)

models = {
    "Logistic Regression": "logistic_regression.pkl",
    "Random Forest": "random_forest.pkl",
    "Decision Tree": "decision_tree.pkl",
    "Tuned Decision Tree": "decision_tree_tuned.pkl",
}

roc_results = {}
pr_results = {}
importance_results = {}

for name, filename in models.items():
    print(f"\nEvaluating {name}...")
    model = joblib.load(MODEL_DIR / filename)

    # Logistic Regression pipeline accepts raw input features.
    # Tree models accept one-hot encoded features.
    if name == "Logistic Regression":
        features = X_test.copy()
        # Ensure numeric inputs match the pipeline's numeric format.
        for col in [
            "step", "amount", "oldbalanceOrg", "newbalanceOrig",
            "oldbalanceDest", "newbalanceDest"
        ]:
            features[col] = pd.to_numeric(
                features[col], errors="coerce"
            )
    else:
        features = pd.get_dummies(
            X_test, columns=["type"], dtype=int
        )
        expected = list(model.feature_names_in_)
        features = features.reindex(
            columns=expected, fill_value=0
        )

    try:
        probabilities = model.predict_proba(features)[:, 1]
        roc_auc = roc_auc_score(y_test, probabilities)
        pr_auc = average_precision_score(y_test, probabilities)

        fpr, tpr, _ = roc_curve(y_test, probabilities)
        precision, recall, _ = precision_recall_curve(
            y_test, probabilities
        )

        roc_results[name] = (fpr, tpr, roc_auc)
        pr_results[name] = (recall, precision, pr_auc)

        print(f"ROC-AUC: {roc_auc:.4f}")
        print(f"PR-AUC: {pr_auc:.4f}")

        if hasattr(model, "feature_importances_"):
            importance = pd.DataFrame({
                "feature": model.feature_names_in_,
                "importance": model.feature_importances_,
            }).sort_values("importance", ascending=False)

            safe_name = name.lower().replace(" ", "_")
            importance.to_csv(
                REPORT_DIR / f"{safe_name}_feature_importance.csv",
                index=False,
            )
            importance_results[name] = importance

    except Exception as error:
        print(f"Could not evaluate {name}: {error}")

# ROC curve
if roc_results:
    plt.figure(figsize=(9, 7))
    for name, (fpr, tpr, auc_value) in roc_results.items():
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc_value:.3f})")

    plt.plot([0, 1], [0, 1], "--", label="Random baseline")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - Fraud Detection")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "roc_curves.png", dpi=150)
    plt.close()

# Precision-Recall curve
if pr_results:
    plt.figure(figsize=(9, 7))
    for name, (recall, precision, ap) in pr_results.items():
        plt.plot(recall, precision, label=f"{name} (AP={ap:.3f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves - Fraud Detection")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "precision_recall_curves.png", dpi=150)
    plt.close()

# Tuned Decision Tree feature importance
if "Tuned Decision Tree" in importance_results:
    top = importance_results["Tuned Decision Tree"].head(15)
    top = top.sort_values("importance")

    plt.figure(figsize=(10, 7))
    plt.barh(top["feature"], top["importance"])
    plt.xlabel("Feature Importance")
    plt.title("Top 15 Features - Tuned Decision Tree")
    plt.tight_layout()
    plt.savefig(
        FIG_DIR / "tuned_decision_tree_feature_importance.png",
        dpi=150,
    )
    plt.close()

print("\nModels evaluated:", list(roc_results.keys()))
print("ROC plot:", FIG_DIR / "roc_curves.png")
print("PR plot:", FIG_DIR / "precision_recall_curves.png")
print("Feature importance CSV files:", REPORT_DIR)
