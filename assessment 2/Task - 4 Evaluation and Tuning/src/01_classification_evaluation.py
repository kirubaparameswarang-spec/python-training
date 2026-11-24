import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)


# -----------------------------------------
# 1. LOAD DATASET
# -----------------------------------------

df = pd.read_csv("data/loan_preprocessing.csv")

print("=" * 55)
print("TASK 4 - CLASSIFICATION MODEL EVALUATION")
print("=" * 55)

TARGET = "loan_status"

X = df.drop(columns=["loan_status", "application_date"])
y = df[TARGET]


# -----------------------------------------
# 2. DEFINE FEATURE COLUMNS
# -----------------------------------------

numeric_features = ["annual_income", "loan_amount", "credit_score"]

categorical_features = [
    "employment_type",
    "education_level",
    "property_area",
    "dependents",
]


# -----------------------------------------
# 3. TRAIN / VALIDATION / TEST SPLIT
# -----------------------------------------

indices = np.arange(len(df))

train_val_idx, test_idx = train_test_split(
    indices, test_size=0.20, random_state=42, stratify=y
)

train_idx, val_idx = train_test_split(
    train_val_idx, test_size=0.25, random_state=42, stratify=y.iloc[train_val_idx]
)

X_train = X.iloc[train_idx]
X_val = X.iloc[val_idx]

y_train = y.iloc[train_idx]
y_val = y.iloc[val_idx]

os.makedirs("outputs", exist_ok=True)

# Save the same split for all later scripts.
np.savez(
    "outputs/data_splits.npz", train_idx=train_idx, val_idx=val_idx, test_idx=test_idx
)

print("\nDATA SPLIT")
print("Training records:", len(train_idx))
print("Validation records:", len(val_idx))
print("Test records:", len(test_idx))


# -----------------------------------------
# 4. PREPROCESSING PIPELINE
# -----------------------------------------

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
# 5. DEFINE CLASSIFICATION MODELS
# -----------------------------------------

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, random_state=42, n_jobs=-1
    ),
    "XGBoost": XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1,
    ),
}


# -----------------------------------------
# 6. TRAIN AND EVALUATE CLASSIFIERS
# -----------------------------------------

results = []
trained_models = {}

print("\nCLASSIFICATION MODEL RESULTS")

for name, classifier in models.items():
    pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", classifier)]
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_val)

    probabilities = pipeline.predict_proba(X_val)[:, 1]

    accuracy = accuracy_score(y_val, predictions)

    precision = precision_score(y_val, predictions, zero_division=0)

    recall = recall_score(y_val, predictions, zero_division=0)

    f1 = f1_score(y_val, predictions, zero_division=0)

    auc = roc_auc_score(y_val, probabilities)

    cm = confusion_matrix(y_val, predictions)

    print("\n" + "-" * 45)
    print("MODEL:", name)
    print("-" * 45)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")

    print("Confusion Matrix:")
    print(cm)

    results.append(
        {
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "ROC_AUC": auc,
            "TN": cm[0, 0],
            "FP": cm[0, 1],
            "FN": cm[1, 0],
            "TP": cm[1, 1],
        }
    )

    trained_models[name] = pipeline


# -----------------------------------------
# 7. SAVE RESULTS AND TRAINED MODELS
# -----------------------------------------

results_df = pd.DataFrame(results)

results_df.to_csv("outputs/classification_metrics.csv", index=False)

joblib.dump(trained_models, "outputs/classification_models.joblib")


# -----------------------------------------
# 8. EXPLAIN ACCURACY AND IMBALANCE
# -----------------------------------------

print("\nWHY ACCURACY CAN BE MISLEADING")

majority_accuracy = y.value_counts(normalize=True).max()

print(f"Majority class baseline accuracy: " f"{majority_accuracy:.4f}")

print(
    "A model can achieve high accuracy by predicting "
    "the majority class while failing to identify "
    "the minority class."
)

print(
    "Precision, recall and F1 provide additional "
    "information about classification performance."
)


print("\nCLASSIFICATION METRICS SAVED:")
print("outputs/classification_metrics.csv")

print("\nTRAINED MODELS SAVED:")
print("outputs/classification_models.joblib")

print("\nDATA SPLITS SAVED:")
print("outputs/data_splits.npz")

print("\nCLASSIFICATION EVALUATION COMPLETED")
