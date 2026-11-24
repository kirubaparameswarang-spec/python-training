import os
import joblib
import numpy as np
import pandas as pd

from scipy.stats import randint, uniform

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone

from xgboost import XGBClassifier


# -----------------------------------------
# 1. LOAD DATA
# -----------------------------------------

df = pd.read_csv("data/loan_preprocessing.csv")

X = df.drop(columns=["loan_status", "application_date"])

y = df["loan_status"]

splits = np.load("outputs/data_splits.npz")

train_idx = splits["train_idx"]

X_train = X.iloc[train_idx]
y_train = y.iloc[train_idx]

os.makedirs("outputs", exist_ok=True)

print("=" * 60)
print("TASK 4 - HYPERPARAMETER TUNING")
print("=" * 60)

print("\nTraining records:", len(X_train))


# -----------------------------------------
# 2. PREPROCESSING
# -----------------------------------------

numeric_features = ["annual_income", "loan_amount", "credit_score"]

categorical_features = [
    "employment_type",
    "education_level",
    "property_area",
    "dependents",
]

numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)


# -----------------------------------------
# 3. CROSS-VALIDATION
# -----------------------------------------

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# -----------------------------------------
# 4. RANDOM FOREST GRID SEARCH CV
# -----------------------------------------

print("\n" + "=" * 60)
print("RANDOM FOREST - GRID SEARCH CV")
print("=" * 60)

rf_pipeline = Pipeline(
    steps=[
        ("preprocessor", clone(preprocessor)),
        ("classifier", RandomForestClassifier(random_state=42, n_jobs=1)),
    ]
)

# Exactly THREE hyperparameters

rf_param_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [None, 5, 10],
    "classifier__min_samples_split": [2, 5],
}

rf_grid_search = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=rf_param_grid,
    scoring="roc_auc",
    cv=cv_strategy,
    n_jobs=-1,
    refit=True,
    verbose=1,
)

rf_grid_search.fit(X_train, y_train)

print("\nRandom Forest Best Parameters:")
print(rf_grid_search.best_params_)

print(
    "\nRandom Forest Best Cross-Validation Score:", round(rf_grid_search.best_score_, 4)
)

pd.DataFrame(rf_grid_search.cv_results_).to_csv(
    "outputs/random_forest_tuning.csv", index=False
)


# -----------------------------------------
# 5. XGBOOST RANDOMIZED SEARCH CV
# -----------------------------------------

print("\n" + "=" * 60)
print("XGBOOST - RANDOMIZED SEARCH CV")
print("=" * 60)

xgb_pipeline = Pipeline(
    steps=[
        ("preprocessor", clone(preprocessor)),
        ("classifier", XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=1)),
    ]
)

xgb_param_distributions = {
    "classifier__n_estimators": randint(100, 401),
    "classifier__max_depth": randint(2, 7),
    "classifier__learning_rate": uniform(0.02, 0.18),
    "classifier__subsample": uniform(0.7, 0.3),
    "classifier__colsample_bytree": uniform(0.7, 0.3),
    "classifier__min_child_weight": randint(1, 6),
}

xgb_random_search = RandomizedSearchCV(
    estimator=xgb_pipeline,
    param_distributions=xgb_param_distributions,
    n_iter=15,
    scoring="roc_auc",
    cv=cv_strategy,
    random_state=42,
    n_jobs=-1,
    refit=True,
    verbose=1,
)

xgb_random_search.fit(X_train, y_train)

print("\nXGBoost Number of Iterations: 15")

print("\nXGBoost Best Parameters:")
print(xgb_random_search.best_params_)

print("\nXGBoost Best Cross-Validation Score:", round(xgb_random_search.best_score_, 4))

pd.DataFrame(xgb_random_search.cv_results_).to_csv(
    "outputs/xgboost_tuning.csv", index=False
)


# -----------------------------------------
# 6. SAVE BEST TUNED MODELS
# -----------------------------------------

tuned_models = {
    "Random Forest Tuned": rf_grid_search.best_estimator_,
    "XGBoost Tuned": xgb_random_search.best_estimator_,
}

joblib.dump(tuned_models, "outputs/tuned_classification_models.joblib")


# -----------------------------------------
# 7. SAVE BEST PARAMETER SUMMARY
# -----------------------------------------

summary = pd.DataFrame(
    [
        {
            "Model": "Random Forest",
            "Best Parameters": str(rf_grid_search.best_params_),
            "Best CV ROC-AUC": rf_grid_search.best_score_,
        },
        {
            "Model": "XGBoost",
            "Best Parameters": str(xgb_random_search.best_params_),
            "Best CV ROC-AUC": xgb_random_search.best_score_,
        },
    ]
)

summary.to_csv("outputs/tuning_summary.csv", index=False)


# -----------------------------------------
# 8. FINAL PRINT
# -----------------------------------------

print("\n" + "=" * 60)
print("HYPERPARAMETER TUNING COMPLETED")
print("=" * 60)

print("\nOUTPUT FILES SAVED:")
print("outputs/random_forest_tuning.csv")
print("outputs/xgboost_tuning.csv")
print("outputs/tuned_classification_models.joblib")
print("outputs/tuning_summary.csv")
