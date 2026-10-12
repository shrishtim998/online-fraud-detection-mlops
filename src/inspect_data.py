
from pathlib import Path
import pandas as pd

# Find the project folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "onlinefraud.csv"

# Load the dataset
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

print("\n--- Dataset Shape ---")
print(df.shape)

print("\n--- First Five Rows ---")
print(df.head())

print("\n--- Column Names ---")
print(df.columns.tolist())

print("\n--- Data Types ---")
print(df.dtypes)

print("\n--- Missing Values ---")
print(df.isnull().sum().sort_values(ascending=False))

print("\n--- Duplicate Rows ---")
print(df.duplicated().sum())

print("\n--- Summary Statistics ---")
print(df.describe(include="all").T)

print("\n--- Fraud Target Distribution ---")
if "isFraud" in df.columns:
    print(df["isFraud"].value_counts(dropna=False))
    print(df["isFraud"].value_counts(normalize=True, dropna=False) * 100)
elif "Is_Fraud" in df.columns:
    print(df["Is_Fraud"].value_counts(dropna=False))
else:
    print("Fraud target column not found. Check the column names.")
