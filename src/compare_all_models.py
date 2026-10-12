
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "reports"

files = {
    "Logistic Regression": "baseline_metrics.json",
    "Random Forest": "random_forest_metrics.json",
    "Decision Tree": "decision_tree_metrics.json",
}

rows = []

for model_name, filename in files.items():
    path = REPORT_DIR / filename

    if not path.exists():
        print(f"Missing metrics file: {path}")
        continue

    with open(path, "r") as f:
        metrics = json.load(f)

    
    rows.append({
        "Model": model_name,
        "Accuracy": metrics.get("accuracy"),
        "Precision": metrics.get("precision"),
        "Recall": metrics.get("recall"),
        "F1 Score": metrics.get("f1_score", metrics.get("f1")),
        "ROC-AUC": metrics.get("roc_auc"),
        "PR-AUC": metrics.get(
            "pr_auc_average_precision",
            metrics.get("average_precision", metrics.get("pr_auc"))
        ),
    })

    

comparison = pd.DataFrame(rows)

numeric_columns = [
    "Accuracy", "Precision", "Recall",
    "F1 Score", "ROC-AUC", "PR-AUC"
]

for column in numeric_columns:
    comparison[column] = pd.to_numeric(
        comparison[column], errors="coerce"
    )

print("\nMODEL COMPARISON")
print(comparison.to_string(
    index=False,
    formatters={
        col: lambda value: f"{value:.4f}"
        for col in numeric_columns
    }
))

output_path = REPORT_DIR / "model_comparison.csv"
comparison.to_csv(output_path, index=False)

print(f"\nComparison saved to: {output_path}")
