# Task 2 - Preprocessing, Features and Pipeline

## Objective
Build a leakage-safe scikit-learn preprocessing and classification pipeline.

## Dataset
`data/loan_preprocessing.csv` is an illustrative synthetic loan application dataset.
It contains numeric, categorical, ordered categorical, date and target columns.
Some values are intentionally missing so the imputation steps can be demonstrated.

## Requirements covered
- Numeric missing values: `SimpleImputer(strategy="median")`
- Categorical missing values: `SimpleImputer(strategy="most_frequent")`
- KNN experiment: `KNNImputer(n_neighbors=5)` on numeric columns
- Nominal encoding: `OneHotEncoder(handle_unknown="ignore")`
- Ordered encoding: `OrdinalEncoder` for education level
- Feature engineering:
  - application date -> year, month and day
  - annual income / loan amount -> income-loan ratio
  - log1p(loan_amount) -> log transform for a skewed feature
- `ColumnTransformer` and all preprocessing steps are inside a scikit-learn `Pipeline`
- 5-fold stratified cross-validation
- Save/load fitted pipeline with `joblib`
- Predict 3 new raw rows after loading the saved pipeline

## Setup (PowerShell)
From this Task 2 folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python .\src\preprocessing_pipeline.py
```

The script prints mean cross-validation accuracy for median and KNN imputation,
saves `outputs/loan_approval_pipeline.joblib`, reloads it, and predicts 3 raw rows.

## Important
The accuracy is data-dependent; do not hard-code the example value 0.84.
The dataset is synthetic and intended for demonstrating preprocessing concepts,
not for real loan decisions.
