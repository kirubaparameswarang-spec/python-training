import os
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 1. LOAD DATA
df = pd.read_csv("data/loan_preprocessing.csv")

# Regression target
y = df["loan_amount"]

# Exclude target and application_date to avoid leakage
X = df.drop(columns=["loan_amount", "application_date", "loan_status"])

# Load saved data split indices
splits = np.load("outputs/data_splits.npz")

train_idx = splits["train_idx"]
val_idx = splits["val_idx"]

X_train = X.iloc[train_idx]
X_val = X.iloc[val_idx]

y_train = y.iloc[train_idx]
y_val = y.iloc[val_idx]

os.makedirs("outputs", exist_ok=True)


print("=" * 60)
print("TASK 4 - REGRESSION EVALUATION")
print("=" * 60)

print("\nTraining records:", len(X_train))
print("Validation records:", len(X_val))


# 2. PREPROCESSING
numeric_features = ["annual_income", "credit_score"]

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


# 3. DEFINE REGRESSION MODELS
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.1, max_iter=10000),
}


# 4. EVALUATE MEAN BASELINE
baseline = DummyRegressor(strategy="mean")
baseline.fit(X_train, y_train)

baseline_predictions = baseline.predict(X_val)

results = []

baseline_mae = mean_absolute_error(y_val, baseline_predictions)
baseline_rmse = np.sqrt(mean_squared_error(y_val, baseline_predictions))
baseline_r2 = r2_score(y_val, baseline_predictions)

results.append(
    {
        "Model": "Mean Baseline",
        "MAE": baseline_mae,
        "RMSE": baseline_rmse,
        "R2": baseline_r2,
    }
)

print("\nMEAN BASELINE")
print("MAE:", round(baseline_mae, 4))
print("RMSE:", round(baseline_rmse, 4))
print("R2:", round(baseline_r2, 4))


# 5. TRAIN AND EVALUATE REGRESSION MODELS
regression_models = {}

for name, model in models.items():
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", model)])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_val)

    mae = mean_absolute_error(y_val, predictions)
    rmse = np.sqrt(mean_squared_error(y_val, predictions))
    r2 = r2_score(y_val, predictions)

    results.append({"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2})

    regression_models[name] = pipeline

    print("\n" + name)
    print("MAE:", round(mae, 4))
    print("RMSE:", round(rmse, 4))
    print("R2:", round(r2, 4))


# 6. SAVE RESULTS
results_df = pd.DataFrame(results)

results_df.to_csv("outputs/regression_metrics.csv", index=False)

joblib.dump(regression_models, "outputs/regression_models.joblib")


# 7. IDENTIFY BEST REGRESSION MODEL
# Lower RMSE indicates better prediction performance.
model_results = results_df[results_df["Model"] != "Mean Baseline"]

best_model = model_results.loc[model_results["RMSE"].idxmin()]

print("\n" + "=" * 60)
print("REGRESSION MODEL COMPARISON")
print("=" * 60)

print(results_df.round(4).to_string(index=False))

print("\nBest Regression Model:", best_model["Model"])
print("Best Validation RMSE:", round(best_model["RMSE"], 4))

print("\nREGRESSION EVALUATION COMPLETED")
print("\nOutput files saved:")
print("outputs/regression_metrics.csv")
print("outputs/regression_models.joblib")
