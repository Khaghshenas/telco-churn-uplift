import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

# Paths for processed data and saved model
PROCESSED_DIR = "data/processed"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "churn_model.joblib")


def load_processed_data(processed_dir: str = PROCESSED_DIR) -> tuple:
    """
    Load preprocessed train/test datasets for churn modeling.

    Parameters
    ----------
    processed_dir : str
        Directory containing processed CSV files (X_train.csv, X_test.csv, y_train.csv, y_test.csv).

    Returns
    -------
    X_train : pd.DataFrame
    X_test : pd.DataFrame
    y_train : pd.Series
    y_test : pd.Series
    
    Notes
    -----
    - Expects CSV files saved by the preprocessing pipeline.
    - y_train and y_test are squeezed into Series for compatibility with scikit-learn.
    """
    X_train = pd.read_csv(os.path.join(processed_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(processed_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(processed_dir, "y_train.csv")).squeeze()
    y_test = pd.read_csv(os.path.join(processed_dir, "y_test.csv")).squeeze()

    return X_train, X_test, y_train, y_test


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """
    Train a Random Forest classifier for churn prediction.

    Parameters
    ----------
    X_train : pd.DataFrame
    y_train : pd.Series

    Returns
    -------
    clf : RandomForestClassifier
        Trained Random Forest model.
    """
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1  # Use all available CPU cores
    )
    clf.fit(X_train, y_train)
    return clf


def evaluate_model(clf: RandomForestClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """
    Evaluate the trained churn model and print metrics.

    Metrics printed:
    - ROC AUC
    - Classification report (precision, recall, f1-score)
    - Confusion matrix

    Parameters
    ----------
    clf : RandomForestClassifier
        Trained model to evaluate.
    X_test : pd.DataFrame
    y_test : pd.Series
    """
    # Predicted probabilities for positive class
    preds_proba = clf.predict_proba(X_test)[:, 1]
    # Predicted labels
    pred_labels = clf.predict(X_test)

    # Compute AUC
    auc = roc_auc_score(y_test, preds_proba)
    
    print("\n=== MODEL EVALUATION ===")
    print(f"ROC AUC: {auc:.4f}\n")

    print("Classification Report:")
    print(classification_report(y_test, pred_labels))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, pred_labels))


def save_model(clf: RandomForestClassifier, model_path: str = MODEL_PATH) -> None:
    """
    Save the trained Random Forest model as a .joblib file.

    Parameters
    ----------
    clf : RandomForestClassifier
        Trained model to save.
    model_path : str
        Path where the model will be saved.
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(clf, model_path)
    print(f"\nModel saved to: {model_path}")


def run_training_pipeline() -> RandomForestClassifier:
    """
    Full churn modeling training pipeline.

    Steps:
    1. Load processed train/test datasets.
    2. Train a Random Forest classifier.
    3. Evaluate the model with ROC AUC, classification report, and confusion matrix.
    4. Save the trained model to disk.

    Returns
    -------
    clf : RandomForestClassifier
        The trained Random Forest model.
    """
    # Step 1: Load processed data
    X_train, X_test, y_train, y_test = load_processed_data()

    # Step 2: Train model
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    # Step 3: Evaluate model
    preds_proba = clf.predict_proba(X_test)[:, 1]
    pred_labels = clf.predict(X_test)
    auc = roc_auc_score(y_test, preds_proba)

    print("\n=== MODEL EVALUATION ===")
    print(f"ROC AUC: {auc:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, pred_labels))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, pred_labels))

    # Step 4: Save model
    save_model(clf)

    return clf


if __name__ == "__main__":
    trained_model = run_training_pipeline()
