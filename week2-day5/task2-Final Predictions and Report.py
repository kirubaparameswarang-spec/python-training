"""
Task 2 - Final Predictions and Report
Customer Churn Prediction

Place customer_churn.csv in this folder before running.
Required columns:
Target: Churn
Numeric: tenure, monthly_charges, total_charges, support_calls, senior_citizen
Categorical: dependents, paperless_billing, contract, internet_service, payment_method
"""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, f1_score, confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "customer_churn.csv"
TARGET = "Churn"
NUMERIC_FEATURES = [
    "tenure", "monthly_charges", "total_charges",
    "support_calls", "senior_citizen"
]
CATEGORICAL_FEATURES = [
    "dependents", "paperless_billing", "contract",
    "internet_service", "payment_method"
]
MODEL_PATH = BASE_DIR / "customer_churn_model.joblib"


def normalize_target(series):
    """Convert common churn labels to 0/1."""
    if pd.api.types.is_numeric_dtype(series):
        values = set(series.dropna().unique())
        if values.issubset({0, 1}):
            return series.astype(int)

    mapping = {
        "yes": 1, "y": 1, "true": 1, "1": 1, "churn": 1,
        "no": 0, "n": 0, "false": 0, "0": 0, "not churn": 0
    }
    converted = series.astype(str).str.strip().str.lower().map(mapping)
    if converted.isna().any():
        bad = series[converted.isna()].dropna().unique().tolist()
        raise ValueError(f"Unrecognized target labels in {TARGET}: {bad}")
    return converted.astype(int)


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Place customer_churn.csv beside this Python file or update DATA_PATH."
        )

    df = pd.read_csv(DATA_PATH)
    required = [TARGET] + NUMERIC_FEATURES + CATEGORICAL_FEATURES
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required dataset columns: {missing}")

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = normalize_target(df[TARGET])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES)
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=42
        ))
    ])

    # Tune using training data only. The held-out test set remains untouched.
    parameter_distributions = {
        "classifier__C": [0.01, 0.1, 0.5, 1, 2, 5, 10, 20],
        "classifier__solver": ["liblinear", "lbfgs"]
    }
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=parameter_distributions,
        n_iter=10,
        scoring="roc_auc",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        random_state=42,
        n_jobs=-1,
        refit=True
    )
    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    # Evaluate exactly once on the held-out test set.
    y_probability = best_model.predict_proba(X_test)[:, 1]
    y_prediction = (y_probability >= 0.5).astype(int)

    test_auc = roc_auc_score(y_test, y_probability)
    test_f1 = f1_score(y_test, y_prediction, zero_division=0)
    cm = confusion_matrix(y_test, y_prediction, labels=[0, 1])

    print("\n========== FINAL TEST RESULTS ==========")
    print(f"Best CV ROC-AUC: {search.best_score_:.4f}")
    print(f"Test ROC-AUC:    {test_auc:.4f}")
    print(f"Test F1-score:   {test_f1:.4f}")
    print("\nConfusion Matrix (rows = actual, columns = predicted):")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_prediction, zero_division=0))
    print("\nBest parameters:")
    print(search.best_params_)

    joblib.dump(best_model, MODEL_PATH)
    print(f"\nSaved model: {MODEL_PATH}")

    # Save final test metrics and confusion matrix.
    metrics = {
        "selected_model": "Tuned Logistic Regression",
        "best_cv_roc_auc": float(search.best_score_),
        "test_roc_auc": float(test_auc),
        "test_f1": float(test_f1),
        "confusion_matrix": cm.tolist(),
        "best_parameters": search.best_params_
    }
    with open(BASE_DIR / "final_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    # Save confusion matrix image.
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm)
    ax.set_title("Customer Churn - Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1], labels=["No Churn", "Churn"])
    ax.set_yticks([0, 1], labels=["No Churn", "Churn"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(BASE_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # Write a one-page-style report with the measured results.
    report = f"""# Customer Churn Prediction - Final Report

## 1. Problem
The objective is to predict whether a customer is likely to leave the service (churn),
so the business can identify at-risk customers and plan retention actions.

## 2. Data
The project uses a synthetic customer dataset with 5,000 rows. The target is
`Churn`. Predictors include tenure, monthly and total charges, support calls,
senior-citizen status, dependents, paperless billing, contract, internet service
and payment method.

## 3. Method
A stratified 80/20 train-test split was used. Numeric missing values were imputed
with the median and scaled using StandardScaler. Categorical missing values were
imputed with the most frequent value and encoded using OneHotEncoder.
Logistic Regression with class balancing was tuned using 5-fold stratified
cross-validation and ROC-AUC. The test set was held out during model selection.

## 4. Results
- Best cross-validation ROC-AUC: {search.best_score_:.4f}
- Held-out test ROC-AUC: {test_auc:.4f}
- Held-out test F1-score (threshold 0.50): {test_f1:.4f}
- Confusion matrix (actual rows; predicted columns):
  - Actual no churn: TN={cm[0,0]}, FP={cm[0,1]}
  - Actual churn: FN={cm[1,0]}, TP={cm[1,1]}

## 5. Limitations
The data is synthetic and may not represent real customer behavior. Model results
depend on the selected probability threshold and the available features.
A model trained on historical patterns may become less accurate as customer
behavior and business conditions change.

## 6. Recommendations
Validate the model on real, recent customer data before operational use. Review
precision and recall alongside ROC-AUC, choose a threshold based on retention
costs, and monitor performance and data drift regularly. Use predictions to
prioritize human-reviewed retention outreach rather than as an automatic decision.

"""
    (BASE_DIR / "final_report.md").write_text(report, encoding="utf-8")
    print("Saved final_report.md, final_metrics.json and confusion_matrix.png")
    print("Next: run predict_new.py with a CSV containing the same feature columns.")


if __name__ == "__main__":
    main()
