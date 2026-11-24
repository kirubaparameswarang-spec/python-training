import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from xgboost import XGBClassifier


# 1. Load dataset
df = pd.read_csv("data/loan_preprocessing.csv")

print("DATASET SHAPE:", df.shape)


# 2. Define features and target
X = df.drop(columns=["loan_status", "application_date"])
y = df["loan_status"]


# 3. Identify numeric and categorical columns
numeric_features = ["annual_income", "loan_amount", "credit_score"]

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


# 5. Split into training and validation data
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# 6. Prepare validation data for XGBoost early stopping
preprocessor_for_validation = preprocessor.fit(X_train)

X_val_processed = preprocessor_for_validation.transform(X_val)


# 7. Define four classification pipelines
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, oob_score=True, random_state=42, n_jobs=-1
    ),
    "XGBoost": XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=3,
        eval_metric="logloss",
        early_stopping_rounds=20,
        random_state=42,
    ),
}


# 8. Train and evaluate all models
results = {}
trained_models = {}

for name, model in models.items():
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])

    print("\nTraining:", name)

    if name == "XGBoost":
        pipeline.fit(
            X_train,
            y_train,
            model__eval_set=[(X_val_processed, y_val)],
            model__verbose=False,
        )
    else:
        pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_val)

    accuracy = accuracy_score(y_val, predictions)

    results[name] = accuracy
    trained_models[name] = pipeline

    print("Validation Accuracy:", round(accuracy, 4))

    if name == "Random Forest":
        print(
            "Random Forest OOB Score:",
            round(pipeline.named_steps["model"].oob_score_, 4),
        )


# 9. Print results table
results_df = pd.DataFrame(list(results.items()), columns=["Model", "Validation score"])

print("\nCLASSIFICATION RESULTS")
print(results_df)

results_df.to_csv("outputs/classification_results.csv", index=False)


# 10. Logistic Regression probability for five rows
logistic_model = trained_models["Logistic Regression"]

print("\nLOGISTIC REGRESSION: FIRST 5 PREDICTIONS")

print("\nPredicted probabilities:")
print(logistic_model.predict_proba(X_val.iloc[:5]))

print("\nPredicted classes:")
print(logistic_model.predict(X_val.iloc[:5]))

print("\nActual classes:")
print(y_val.iloc[:5].to_numpy())


# 11. Get transformed feature names
feature_names = (
    trained_models["Random Forest"].named_steps["preprocessor"].get_feature_names_out()
)


# 12. Random Forest top 5 important features
forest_importances = (
    trained_models["Random Forest"].named_steps["model"].feature_importances_
)

forest_top5 = (
    pd.Series(forest_importances, index=feature_names)
    .sort_values(ascending=False)
    .head(5)
)

print("\nRANDOM FOREST TOP 5 FEATURES")
print(forest_top5)


# 13. XGBoost top 5 important features
xgb_importances = trained_models["XGBoost"].named_steps["model"].feature_importances_

xgb_top5 = (
    pd.Series(xgb_importances, index=feature_names).sort_values(ascending=False).head(5)
)

print("\nXGBOOST TOP 5 FEATURES")
print(xgb_top5)


# 14. Classification report for best model
best_model_name = max(results, key=results.get)

print("\nBEST MODEL:", best_model_name)

print(classification_report(y_val, trained_models[best_model_name].predict(X_val)))

print("\nCLASSIFICATION TASK COMPLETED")
