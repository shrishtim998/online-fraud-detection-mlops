
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "onlinefraud.csv"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

# 1. Target distribution
print("\nFraud distribution:")
print(df["isFraud"].value_counts())
print("\nFraud percentage:")
print((df["isFraud"].value_counts(normalize=True) * 100).round(4))

plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="isFraud")
plt.title("Legitimate vs Fraudulent Transactions")
plt.xlabel("isFraud (0 = Legitimate, 1 = Fraud)")
plt.ylabel("Number of Transactions")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fraud_distribution.png", dpi=150)
plt.close()

# 2. Transaction types
print("\nFraud rate by transaction type:")
print(
    df.groupby("type")["isFraud"]
    .agg(["count", "sum", "mean"])
    .assign(fraud_rate_percent=lambda x: x["mean"] * 100)
    .drop(columns="mean")
    .sort_values("fraud_rate_percent", ascending=False)
)

plt.figure(figsize=(9, 5))
sns.countplot(data=df, x="type", hue="isFraud")
plt.title("Transaction Types by Fraud Label")
plt.xlabel("Transaction Type")
plt.ylabel("Count")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "transaction_types.png", dpi=150)
plt.close()

# 3. Transaction amount distribution
plt.figure(figsize=(8, 5))
sns.histplot(
    data=df,
    x="amount",
    hue="isFraud",
    bins=60,
    element="step",
    stat="density",
    common_norm=False
)
plt.xscale("log")
plt.title("Transaction Amount Distribution (Log Scale)")
plt.xlabel("Amount (log scale)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "amount_distribution.png", dpi=150)
plt.close()

# 4. Numeric correlations
numeric_df = df.select_dtypes(include="number")
corr = numeric_df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr, cmap="coolwarm", center=0, annot=True, fmt=".2f")
plt.title("Numeric Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=150)
plt.close()

print("\nEDA completed.")
print(f"Charts saved to: {OUTPUT_DIR}")
