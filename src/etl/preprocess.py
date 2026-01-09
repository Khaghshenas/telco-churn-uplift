import os
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

# Paths relative to the project root
RAW_DATA_PATH = "../data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
PROCESSED_DIR = "data/processed"
TARGET_COL = "Churn"


def load_raw_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load the raw Telco Customer Churn dataset.

    Parameters
    ----------
    path : str
        Path to the CSV file containing the raw dataset.

    Returns
    -------
    pd.DataFrame
        Raw dataset as a DataFrame.
    """
    return pd.read_csv(path)


def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw dataset and create basic engineered features.

    Steps:
    1. Strip leading/trailing spaces in object columns.
    2. Convert 'TotalCharges' to numeric; drop rows with missing values.
    3. Create 'tenure_group' as a categorical feature.
    4. Map 'Churn' from 'No'/'Yes' to 0/1.
    5. Drop 'customerID', which is an identifier and not a feature.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset.

    Returns
    -------
    pd.DataFrame
        Cleaned and feature-engineered dataset ready for modeling.
    """
    df = df.copy()

    # Strip whitespace from string columns
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()

    # Convert TotalCharges to numeric; drop rows where conversion fails
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"])

    # Create tenure groups
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=["0-12", "12-24", "24-48", "48-72"],
        include_lowest=True,
    )

    # Convert target column to 0/1
    df[TARGET_COL] = df[TARGET_COL].map({"No": 0, "Yes": 1})

    # Drop customerID as it is not a predictive feature
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    return df


import os
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

PROCESSED_DIR = "data/processed"
TARGET_COL = "Churn"


def encode_features(df: pd.DataFrame, target_col: str = TARGET_COL) -> Tuple[pd.DataFrame, pd.Series]:
    """
    One-hot encode categorical features and separate features (X) from target (y).

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed dataset including both features and target.
    target_col : str
        Name of the target column.

    Returns
    -------
    X_encoded : pd.DataFrame
        Feature matrix with categorical variables one-hot encoded.
    y : pd.Series
        Target vector.
    
    Notes
    -----
    - Target column is separated as y.
    - All remaining non-numeric columns are one-hot encoded.
    - Drop first level to avoid dummy variable trap.
    """
    df = df.copy()

    # Separate target from features
    y = df[target_col]
    X = df.drop(columns=[target_col])

    # One-hot encode categorical features
    X_encoded = pd.get_dummies(X, drop_first=True)

    return X_encoded, y


def split_and_save(
    X: pd.DataFrame,
    y: pd.Series,
    out_dir: str = PROCESSED_DIR,
    test_size: float = 0.2,
    random_state: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split features and target into train and test sets, and save them to CSV.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target vector.
    out_dir : str
        Directory where processed CSVs will be saved.
    test_size : float
        Fraction of data to use as test set.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    X_train, X_test, y_train, y_test : tuple
        Train/test splits of features and target.

    Notes
    -----
    - Stratified split ensures churn ratio is maintained in train/test sets.
    - Feature column names are saved to a JSON file to preserve column order for serving or prediction pipelines.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,  # preserve churn class ratio
    )

    # Save CSVs
    X_train.to_csv(os.path.join(out_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(out_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(out_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(out_dir, "y_test.csv"), index=False)

    # Save feature column names for consistent ordering during serving
    feature_cols_path = os.path.join(out_dir, "feature_columns.json")
    X_train.columns.to_series().to_json(feature_cols_path, orient="values")

    return X_train, X_test, y_train, y_test

def run_full_preprocess() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Run the full preprocessing pipeline from raw CSV to train/test CSV files.

    Steps:
    1. Load raw Telco Customer Churn data.
    2. Clean and engineer features (handle missing values, create tenure groups, map target).
    3. Encode categorical features using one-hot encoding.
    4. Split into train/test sets and save to CSV.

    Returns
    -------
    X_train, X_test, y_train, y_test : tuple
        Train/test splits of features and target, saved to disk for downstream modeling.
    """
    # Step 1: Load raw data
    df_raw = load_raw_data()

    # Step 2: Clean and engineer features
    df_clean = clean_and_engineer(df_raw)

    # Step 3: Encode categorical features
    X, y = encode_features(df_clean)

    # Step 4: Split into train/test and save to CSV
    return split_and_save(X, y)


if __name__ == "__main__":
    
    X_train, X_test, y_train, y_test = run_full_preprocess()
    print("Preprocessing complete. Train/test CSVs saved in 'data/processed'.")

