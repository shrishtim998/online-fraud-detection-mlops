
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "reports"

model_files = {
    "Logistic Regression": "baseline_metrics.json",
    "Random Forest": "random_forest_metrics.json",
    "Decision Tree": "decision_tree_metrics.json",
    "Tuned Decision Tree": "decision_tree_tuning_metrics.json",
}

rows = []

for model_name, filename in model_files.items():
    path = REPORT_DIR / filename

    if not path.exists():
        print(f"Skipping missing file: {filename}")
        continue

    with open(path, "r") as file:
        metrics = json.load(file)

    # Tuned Decision Tree uses test_* metric names.
    prefix = "test_" if model_name == "Tuned Decision Tree" else ""

    precision = metrics.get(
        f"{prefix}precision", metrics.get("precision")
    )
    recall = metrics.get(
        f"{prefix}recall", metrics.get("recall")
    )
    f1 = metrics.get(
        f"{prefix}f1",
        metrics.get("f1_score", metrics.get("f1"))
    )
    roc_auc = metrics.get(
        f"{prefix}roc_auc", metrics.get("roc_auc")
    )
    pr_auc = metrics.get(
        f"{prefix}pr_auc",
        metrics.get(
            "pr_auc_average_precision",
            metrics.get("average_precision", metrics.get("pr_auc"))
        )
    )
    accuracy = metrics.get(
        f"{prefix}accuracy", metrics.get("accuracy")
    )

    rows.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc,
    })

comparison = pd.DataFrame(rows)

metric_columns = [
    "Accuracy", "Precision", "Recall",
    "F1 Score", "ROC-AUC", "PR-AUC"
]

for col in metric_columns:
    comparison[col] = pd.to_numeric(
        comparison[col], errors="coerce"
    )

output = REPORT_DIR / "final_model_comparison.csv"
comparison.to_csv(output, index=False)

print("\nFINAL MODEL COMPARISON")
print(comparison.to_string(
    index=False,
    formatters={
        col: lambda x: f"{x:.4f}" if pd.notna(x) else "MISSING"
        for col in metric_columns
    }
))

print(f"\nSaved: {output}")

if comparison[["F1 Score", "PR-AUC"]].isna().any().any():
    print(
        "\nWARNING: Some metrics are missing. "
        "Check the corresponding JSON files before selecting a model."
    )
else:
    print("\nAll comparison metrics were loaded.")
