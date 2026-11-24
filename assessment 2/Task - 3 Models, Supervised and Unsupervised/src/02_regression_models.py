import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_absolute_error


# 1. Load dataset
df = pd.read_csv("data/loan_preprocessing.csv")


# 2. Define features and target
X = df.drop(columns=["loan_amount", "loan_status", "application_date"])

y = df["loan_amount"]


# 3. Define numeric and categorical features
numeric_features = ["annual_income", "credit_score"]

categorical_features = [
    "employment_type",
    "education_level",
    "property_area",
    "dependents",
]


# 4. Create preprocessing pipelines
numeric_pipeline = Pipeline(
    [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)

categorical_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    [
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ]
)


# 5. Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# 6. Define regression models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=1000, max_iter=10000),
}


# 7. Train and evaluate models
results = {}
coefficients = {}

for name, model in models.items():
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])

    print("\nTraining:", name)

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_val)

    r2 = r2_score(y_val, predictions)
    mae = mean_absolute_error(y_val, predictions)

    results[name] = {"R2 Score": r2, "MAE": mae}

    print("R2 Score:", round(r2, 4))
    print("Mean Absolute Error:", round(mae, 2))

    # Get feature names
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()

    # Get model coefficients
    model_coefficients = pipeline.named_steps["model"].coef_

    coefficients[name] = pd.Series(model_coefficients, index=feature_names)


# 8. Print regression results
results_df = pd.DataFrame(results).T

print("\nREGRESSION RESULTS")
print(results_df)

results_df.to_csv("outputs/regression_results.csv")


# 9. Compare coefficients
coefficients_df = pd.DataFrame(coefficients)

print("\nCOEFFICIENT COMPARISON")
print(coefficients_df)

coefficients_df.to_csv("outputs/regression_coefficients.csv")


# 10. Count zero coefficients in Lasso
lasso_coefficients = coefficients["Lasso Regression"]

zero_count = (lasso_coefficients.abs() < 1e-6).sum()

print("\nLASSO ZERO COEFFICIENT COUNT:", zero_count)

print("TOTAL FEATURES:", len(lasso_coefficients))

print("\nLASSO SELECTED FEATURES")
print(lasso_coefficients[lasso_coefficients.abs() >= 1e-6])


print("\nREGRESSION TASK COMPLETED")
