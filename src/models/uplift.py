import os
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklift.metrics import uplift_at_k, qini_auc_score

PROCESSED_DIR = "data/processed"
MODEL_DIR = "models"


def assign_synthetic_treatment(X: pd.DataFrame, seed: int = 42):
    """Randomly assign treatment=1 or control=0 for demonstration."""
    np.random.seed(seed)
    treatment = np.random.binomial(1, 0.5, size=len(X))
    X_with_treat = X.copy()
    X_with_treat["treatment"] = treatment
    return X_with_treat, treatment


def split_treated_control(X: pd.DataFrame, y: pd.Series):
    """Split dataset into treated and control groups."""
    treated_idx = X[X["treatment"] == 1].index
    control_idx = X[X["treatment"] == 0].index

    X_t = X.loc[treated_idx].drop(columns=["treatment"])
    y_t = y.loc[treated_idx]

    X_c = X.loc[control_idx].drop(columns=["treatment"])
    y_c = y.loc[control_idx]

    return X_t, y_t, X_c, y_c


def train_uplift_models(X_train, y_train):
    """Train two separate models: treated model and control model."""
    X_train_t, y_train_t, X_train_c, y_train_c = split_treated_control(X_train, y_train)

    model_t = GradientBoostingClassifier(n_estimators=150, random_state=42)
    model_c = GradientBoostingClassifier(n_estimators=150, random_state=42)

    model_t.fit(X_train_t, y_train_t)
    model_c.fit(X_train_c, y_train_c)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model_t, os.path.join(MODEL_DIR, "uplift_model_t.joblib"))
    joblib.dump(model_c, os.path.join(MODEL_DIR, "uplift_model_c.joblib"))

    print("Uplift models saved.")

    return model_t, model_c


def predict_uplift(model_t, model_c, X_test):
    """Compute uplift: P(churn|treated) − P(churn|control)."""
    prob_t = model_t.predict_proba(X_test)[:, 1]
    prob_c = model_c.predict_proba(X_test)[:, 1]

    uplift = prob_t - prob_c
    return uplift, prob_t, prob_c

def evaluate_uplift(y_true, uplift_scores, treatment, top_k=0.2):
    """
    Evaluate uplift model performance.

    Parameters
    ----------
    y_true : array-like
        True outcome labels (0/1)
    uplift_scores : array-like
        Predicted uplift scores
    treatment : array-like
        Treatment assignment (0=control, 1=treated)
    top_k : float
        Fraction of top scored customers to compute uplift@k

    Returns
    -------
    dict
        Metrics: 'uplift_at_k' and 'qini_auc'
    """
    metrics = {}
    metrics["uplift_at_{}_pct".format(int(top_k*100))] = uplift_at_k(
        y_true, uplift_scores, treatment, strategy="by_group", k=top_k
    )
    metrics["qini_auc"] = qini_auc_score(y_true, uplift_scores, treatment)
    return metrics

def run_uplift_pipeline():
    """Load processed data, generate synthetic treatment, train models, compute uplift."""
    X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train.csv")).squeeze()
    y_test = pd.read_csv(os.path.join(PROCESSED_DIR, "y_test.csv")).squeeze()

    # Step 1: Assign synthetic treatment
    X_train_treat, treat_train = assign_synthetic_treatment(X_train)
    X_test_treat, treat_test = assign_synthetic_treatment(X_test)

    # Step 2: Train uplift models
    model_t, model_c = train_uplift_models(X_train_treat, y_train)

    # Step 3: Predict uplift on test
    uplift, p_t, p_c = predict_uplift(model_t, model_c, X_test_treat.drop(columns=["treatment"]))

    print("\nSample uplift scores:")
    print(uplift[:10])

    # Step 4: Evaluate uplift
    metrics = evaluate_uplift(y_test, uplift, treat_test)
    print("\nUplift Evaluation Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    return uplift, p_t, p_c, metrics
