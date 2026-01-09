import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklift.metrics import uplift_at_k, qini_auc_score

# Paths for processed data and models
PROCESSED_DIR = "data/processed"
MODEL_DIR = "models"


def assign_synthetic_treatment(X: pd.DataFrame, seed: int = 42) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Randomly assign a synthetic treatment for uplift modeling.

    This function is used for demonstration purposes to create a treatment column.
    In real-world scenarios, the treatment column should come from experimental or observational data.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix without treatment assignment.
    seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    X_with_treatment : pd.DataFrame
        Copy of the input DataFrame with an added 'treatment' column (0=control, 1=treated).
    treatment : np.ndarray
        Array of treatment assignments.
    """
    np.random.seed(seed)
    treatment = np.random.binomial(1, 0.5, size=len(X))
    X_with_treatment = X.copy()
    X_with_treatment["treatment"] = treatment
    return X_with_treatment, treatment


def split_treated_control(X: pd.DataFrame, y: pd.Series) -> tuple:
    """
    Split the dataset into treated and control groups for uplift modeling.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix, must include a 'treatment' column (0=control, 1=treated).
    y : pd.Series
        Target vector.

    Returns
    -------
    X_t : pd.DataFrame
        Features of treated group.
    y_t : pd.Series
        Target of treated group.
    X_c : pd.DataFrame
        Features of control group.
    y_c : pd.Series
        Target of control group.
    """
    treated_idx = X[X["treatment"] == 1].index
    control_idx = X[X["treatment"] == 0].index

    X_t = X.loc[treated_idx].drop(columns=["treatment"])
    y_t = y.loc[treated_idx]

    X_c = X.loc[control_idx].drop(columns=["treatment"])
    y_c = y.loc[control_idx]

    return X_t, y_t, X_c, y_c


def train_uplift_models(X_train: pd.DataFrame, y_train: pd.Series) -> tuple:
    """
    Train two separate Gradient Boosting models: one for the treated group and one for the control group.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features including 'treatment' column.
    y_train : pd.Series
        Training target vector.

    Returns
    -------
    model_t : GradientBoostingClassifier
        Model trained on treated group.
    model_c : GradientBoostingClassifier
        Model trained on control group.

    Notes
    -----
    - The models are saved to disk in the 'models/' directory as 'uplift_model_t.joblib'
      and 'uplift_model_c.joblib'.
    """
    # Split dataset
    X_train_t, y_train_t, X_train_c, y_train_c = split_treated_control(X_train, y_train)

    # Initialize models
    model_t = GradientBoostingClassifier(n_estimators=150, random_state=42)
    model_c = GradientBoostingClassifier(n_estimators=150, random_state=42)

    # Train models
    model_t.fit(X_train_t, y_train_t)
    model_c.fit(X_train_c, y_train_c)

    # Save models
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model_t, os.path.join(MODEL_DIR, "uplift_model_t.joblib"))
    joblib.dump(model_c, os.path.join(MODEL_DIR, "uplift_model_c.joblib"))

    print("Uplift models saved.")

    return model_t, model_c


def predict_uplift(
    model_t: GradientBoostingClassifier,
    model_c: GradientBoostingClassifier,
    X_test: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute uplift scores for a test set.

    Uplift is calculated as the difference in predicted probabilities:
        P(outcome | treated) − P(outcome | control)

    Parameters
    ----------
    model_t : GradientBoostingClassifier
        Model trained on treated group.
    model_c : GradientBoostingClassifier
        Model trained on control group.
    X_test : pd.DataFrame
        Test features (without 'treatment' column).

    Returns
    -------
    uplift : np.ndarray
        Predicted uplift scores.
    prob_t : np.ndarray
        Predicted probabilities from the treated model.
    prob_c : np.ndarray
        Predicted probabilities from the control model.
    """
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


def run_uplift_pipeline() -> tuple[pd.Series, pd.Series, pd.Series, dict]:
    """
    Full uplift modeling pipeline: load data, assign treatment, train models, predict uplift, and evaluate.

    Steps
    -----
    1. Load preprocessed train/test datasets.
    2. Assign synthetic treatment for demonstration purposes.
    3. Train Gradient Boosting models for treated and control groups.
    4. Predict uplift scores on the test set.
    5. Evaluate uplift performance using uplift@k and Qini AUC.

    Returns
    -------
    uplift : pd.Series
        Predicted uplift scores for the test set.
    prob_t : pd.Series
        Predicted probabilities from the treated model.
    prob_c : pd.Series
        Predicted probabilities from the control model.
    metrics : dict
        Dictionary with evaluation metrics ('uplift_at_20_pct', 'qini_auc').
    """
    # Step 0: Load processed data
    X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train.csv")).squeeze()
    y_test = pd.read_csv(os.path.join(PROCESSED_DIR, "y_test.csv")).squeeze()

    # Step 1: Assign synthetic treatment
    X_train_treat, treat_train = assign_synthetic_treatment(X_train)
    X_test_treat, treat_test = assign_synthetic_treatment(X_test)

    # Step 2: Train uplift models
    model_t, model_c = train_uplift_models(X_train_treat, y_train)

    # Step 3: Predict uplift on test set
    uplift, prob_t, prob_c = predict_uplift(
        model_t, model_c, X_test_treat.drop(columns=["treatment"])
    )

    print("\nSample uplift scores:")
    print(uplift[:10])

    # Step 4: Evaluate uplift
    metrics = evaluate_uplift(y_test.values, uplift, treat_test)
    print("\n=== UPLIFT EVALUATION METRICS ===")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")

    return uplift, prob_t, prob_c, metrics


if __name__ == "__main__":
    run_uplift_pipeline()
