import os
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

# Paths relative to your project root
RAW_DATA_PATH = "../data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
PROCESSED_DIR = "data/processed"
TARGET_COL = "Churn"


def load_raw_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw Telco Customer Churn dataset."""
    df = pd.read_csv(path)
    return df


def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw dataframe and create basic engineered features.

    Steps:
    - Strip whitespace in object columns
    - Convert TotalCharges to numeric and drop rows where it's missing
    - Create a tenure_group feature
    - Drop customerID (identifier, not a feature)
    - Map Churn from 'No'/'Yes' to 0/1
    """
    df = df.copy()

    # Strip spaces in object columns (handles ' ' as missing)
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()

    # Convert TotalCharges to numeric, coerce errors to NaN, drop those rows
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"])

    # Tenure group (example feature)
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=["0-12", "12-24", "24-48", "48-72"],
        include_lowest=True,
    )

    # Map Churn to 0/1
    df[TARGET_COL] = df[TARGET_COL].map({"No": 0, "Yes": 1})

    # Drop customerID (you can keep it elsewhere if you want)
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    return df


def encode_features(df: pd.DataFrame, target_col: str = TARGET_COL) -> Tuple[pd.DataFrame, pd.Series]:
    """One-hot encode categorical features and separate X, y.

    - Target column is kept as y
    - All remaining non-numeric columns are one-hot encoded
    """
    df = df.copy()

    y = df[target_col]
    X = df.drop(columns=[target_col])

    # One-hot encode categoricals
    X_encoded = pd.get_dummies(X, drop_first=True)

    return X_encoded, y


def split_and_save(
    X: pd.DataFrame,
    y: pd.Series,
    out_dir: str = PROCESSED_DIR,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Split into train/test sets and save to CSV files."""
    os.makedirs(out_dir, exist_ok=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,  # keep churn ratio similar in train/test
    )

    X_train.to_csv(os.path.join(out_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(out_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(out_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(out_dir, "y_test.csv"), index=False)

    # Save feature columns so the serving code can recreate the same order
    feature_cols_path = os.path.join(out_dir, "feature_columns.json")
    X_train.columns.to_series().to_json(feature_cols_path, orient="values")

    return X_train, X_test, y_train, y_test


def run_full_preprocess():
    """Run the full ETL pipeline from raw CSV to processed train/test CSVs."""
    df_raw = load_raw_data()
    df_clean = clean_and_engineer(df_raw)
    X, y = encode_features(df_clean)
    return split_and_save(X, y)


if __name__ == "__main__":
    run_full_preprocess()

