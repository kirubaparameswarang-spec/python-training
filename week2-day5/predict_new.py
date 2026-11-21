"""
Predict churn for new customers.

Usage:
    python predict_new.py new_customers.csv

The input CSV must contain the same feature columns used for training.
Output: predictions.csv in the same folder.
"""

from pathlib import Path
import sys
import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "customer_churn_model.joblib"
OUTPUT_PATH = BASE_DIR / "predictions.csv"

FEATURE_COLUMNS = [
    "tenure", "monthly_charges", "total_charges", "support_calls",
    "senior_citizen", "dependents", "paperless_billing", "contract",
    "internet_service", "payment_method"
]


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python predict_new.py new_customers.csv")

    input_path = Path(sys.argv[1])
    if not input_path.is_absolute():
        input_path = BASE_DIR / input_path

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model not found. Run task2-Final Predictions and Report.py first."
        )
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    new_customers = pd.read_csv(input_path)
    missing = [c for c in FEATURE_COLUMNS if c not in new_customers.columns]
    if missing:
        raise ValueError(f"Input CSV is missing columns: {missing}")

    model = joblib.load(MODEL_PATH)
    probabilities = model.predict_proba(new_customers[FEATURE_COLUMNS])[:, 1]
    labels = (probabilities >= 0.5).astype(int)

    results = new_customers.copy()
    results["churn_probability"] = probabilities
    results["predicted_churn"] = labels
    results["predicted_churn_label"] = results["predicted_churn"].map(
        {0: "No Churn", 1: "Churn"}
    )
    results.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved predictions to: {OUTPUT_PATH}")
    print(results[["churn_probability", "predicted_churn_label"]].head())


if __name__ == "__main__":
    main()
