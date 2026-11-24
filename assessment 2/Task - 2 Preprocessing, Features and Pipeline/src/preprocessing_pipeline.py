from pathlib import Path
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    FunctionTransformer,
)
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "loan_preprocessing.csv"
OUTPUT_DIR = ROOT / "outputs"
MODEL_PATH = OUTPUT_DIR / "loan_approval_pipeline.joblib"

TARGET = "loan_status"
NUMERIC_FEATURES = [
    "annual_income",
    "loan_amount",
    "credit_score",
    "application_year",
    "application_month",
    "application_day",
    "income_loan_ratio",
    "log_loan_amount",
]
CATEGORICAL_FEATURES = ["employment_type", "property_area", "dependents"]
ORDINAL_FEATURES = ["education_level"]
EDUCATION_ORDER = [["High School", "Graduate", "Postgraduate"]]


def add_engineered_features(data):
    """Create date, interaction, and log features without learning from the data."""
    frame = data.copy()
    date = pd.to_datetime(frame["application_date"], errors="coerce")
    frame["application_year"] = date.dt.year
    frame["application_month"] = date.dt.month
    frame["application_day"] = date.dt.day
    frame["income_loan_ratio"] = pd.to_numeric(
        frame["annual_income"], errors="coerce"
    ) / pd.to_numeric(frame["loan_amount"], errors="coerce").replace(0, np.nan)
    # Loan amount is right-skewed; log1p reduces its skew.
    frame["log_loan_amount"] = np.log1p(
        pd.to_numeric(frame["loan_amount"], errors="coerce").clip(lower=0)
    )
    return frame


def make_preprocessor(numeric_imputer):
    numeric_pipeline = Pipeline(
        [("imputer", numeric_imputer), ("scaler", StandardScaler())]
    )

    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    ordinal_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "ordinal",
                OrdinalEncoder(
                    categories=EDUCATION_ORDER,
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
            ("ordinal", ordinal_pipeline, ORDINAL_FEATURES),
        ],
        remainder="drop",
    )


def make_pipeline(numeric_imputer):
    return Pipeline(
        [
            (
                "feature_engineering",
                FunctionTransformer(add_engineered_features, validate=False),
            ),
            ("preprocessor", make_preprocessor(numeric_imputer)),
            ("classifier", LogisticRegression(max_iter=2000, random_state=42)),
        ]
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(DATA_PATH)

    X = data.drop(columns=[TARGET])
    y = data[TARGET].astype(int)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Required main experiment: median imputation for numeric columns.
    median_pipeline = make_pipeline(SimpleImputer(strategy="median"))
    median_scores = cross_val_score(
        median_pipeline, X, y, cv=cv, scoring="accuracy", n_jobs=-1
    )
    print(f"Pipeline CV accuracy (median): {median_scores.mean():.2f}")

    # Try KNNImputer on numeric columns as a comparison.
    knn_pipeline = make_pipeline(KNNImputer(n_neighbors=5))
    knn_scores = cross_val_score(
        knn_pipeline, X, y, cv=cv, scoring="accuracy", n_jobs=-1
    )
    print(f"Pipeline CV accuracy (KNN):    {knn_scores.mean():.2f}")

    # Fit the selected median-imputation pipeline on all available training data.
    # All imputation, encoding, scaling and feature engineering remain in the pipeline.
    final_pipeline = median_pipeline
    final_pipeline.fit(X, y)
    joblib.dump(final_pipeline, MODEL_PATH)
    print(f"Saved fitted pipeline: {MODEL_PATH}")

    # Load the saved model and predict from 3 raw rows (no manual preprocessing).
    loaded_pipeline = joblib.load(MODEL_PATH)
    new_raw_rows = pd.DataFrame(
        [
            {
                "application_date": "2026-01-15",
                "annual_income": 85000,
                "loan_amount": 250000,
                "credit_score": 740,
                "employment_type": "Salaried",
                "education_level": "Graduate",
                "property_area": "Urban",
                "dependents": "0",
            },
            {
                "application_date": "2026-02-20",
                "annual_income": np.nan,
                "loan_amount": 600000,
                "credit_score": 610,
                "employment_type": "Self-employed",
                "education_level": "High School",
                "property_area": "Rural",
                "dependents": "2",
            },
            {
                "application_date": "2026-03-10",
                "annual_income": 140000,
                "loan_amount": 400000,
                "credit_score": 800,
                "employment_type": "Business",
                "education_level": "Postgraduate",
                "property_area": "Semiurban",
                "dependents": "1",
            },
        ]
    )

    predictions = loaded_pipeline.predict(new_raw_rows)
    print("Predictions for 3 new rows:", predictions.tolist())
    print("Class meaning: 0 = Not Approved, 1 = Approved")


if __name__ == "__main__":
    main()
