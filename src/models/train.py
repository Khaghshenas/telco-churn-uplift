import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix
)

PROCESSED_DIR = "data/processed"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "churn_model.joblib")


def load_processed_data(processed_dir: str = PROCESSED_DIR):
    """Load train/test splits created by preprocess.py."""
    X_train = pd.read_csv(os.path.join(processed_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(processed_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(processed_dir, "y_train.csv")).squeeze()
    y_test = pd.read_csv(os.path.join(processed_dir, "y_test.csv")).squeeze()

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """Train a RandomForest churn classifier."""
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    return clf


def evaluate_model(clf, X_test, y_test):
    """Print AUC, classification report, confusion matrix."""
    preds = clf.predict_proba(X_test)[:, 1]
    pred_labels = clf.predict(X_test)

    auc = roc_auc_score(y_test, preds)
    print("\n=== MODEL EVALUATION ===")
    print(f"AUC: {auc:.4f}\n")

    print("Classification Report:")
    print(classification_report(y_test, pred_labels))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, pred_labels))


def save_model(clf, model_path: str = MODEL_PATH):
    """Save trained model as a .joblib file."""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(clf, model_path)
    print(f"\nModel saved to: {model_path}")


def run_training_pipeline():
    """Full training pipeline."""
    X_train, X_test, y_train, y_test = load_processed_data()
    clf = train_model(X_train, y_train)
    evaluate_model(clf, X_test, y_test)
    save_model(clf)
    return clf


if __name__ == "__main__":
    run_training_pipeline()
