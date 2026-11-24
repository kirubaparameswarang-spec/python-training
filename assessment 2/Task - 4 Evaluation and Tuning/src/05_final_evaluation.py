import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    precision_recall_curve,
)


# ==========================================================
# 1. LOAD DATA AND SAVED SPLITS
# ==========================================================

df = pd.read_csv("data/loan_preprocessing.csv")

X = df.drop(columns=["loan_status", "application_date"])
y = df["loan_status"]

splits = np.load("outputs/data_splits.npz")

train_idx = splits["train_idx"]
val_idx = splits["val_idx"]
test_idx = splits["test_idx"]

X_train = X.iloc[train_idx]
y_train = y.iloc[train_idx]

X_val = X.iloc[val_idx]
y_val = y.iloc[val_idx]

X_test = X.iloc[test_idx]
y_test = y.iloc[test_idx]

print("=" * 65)
print("TASK 4 - FINAL CLASSIFICATION EVALUATION")
print("=" * 65)

print("\nTraining records:", len(X_train))
print("Validation records:", len(X_val))
print("Test records:", len(X_test))


# ==========================================================
# 2. LOAD BASELINE AND TUNED MODELS
# ==========================================================

baseline_models = joblib.load("outputs/classification_models.joblib")

tuned_models = joblib.load("outputs/tuned_classification_models.joblib")

candidate_models = {
    "Logistic Regression": baseline_models["Logistic Regression"],
    "Decision Tree": baseline_models["Decision Tree"],
    "Random Forest": baseline_models["Random Forest"],
    "XGBoost": baseline_models["XGBoost"],
    "Random Forest Tuned": tuned_models["Random Forest Tuned"],
    "XGBoost Tuned": tuned_models["XGBoost Tuned"],
}


# ==========================================================
# 3. VALIDATION MODEL COMPARISON
# ==========================================================

validation_results = []
validation_predictions = {}

for name, model in candidate_models.items():
    probabilities = model.predict_proba(X_val)[:, 1]

    predictions = (probabilities >= 0.5).astype(int)

    auc = roc_auc_score(y_val, probabilities)
    f1 = f1_score(y_val, predictions, zero_division=0)

    validation_results.append(
        {"Model": name, "Validation ROC-AUC": auc, "Validation F1": f1}
    )

    validation_predictions[name] = probabilities

validation_df = pd.DataFrame(validation_results)

validation_df = validation_df.sort_values(
    by=["Validation ROC-AUC", "Validation F1"], ascending=False
)

validation_df.to_csv("outputs/validation_model_comparison.csv", index=False)

print("\nVALIDATION MODEL COMPARISON")
print(validation_df.round(4).to_string(index=False))


# ==========================================================
# 4. SELECT BEST MODEL
# ==========================================================

best_model_name = validation_df.iloc[0]["Model"]

best_model = candidate_models[best_model_name]

best_validation_auc = validation_df.iloc[0]["Validation ROC-AUC"]
best_validation_f1 = validation_df.iloc[0]["Validation F1"]

print("\nSelected Best Model:", best_model_name)
print("Validation ROC-AUC:", round(best_validation_auc, 4))
print("Validation F1:", round(best_validation_f1, 4))


# ==========================================================
# 5. SELECT THRESHOLD USING VALIDATION DATA
# ==========================================================

val_probabilities = validation_predictions[best_model_name]

precision_values, recall_values, thresholds = precision_recall_curve(
    y_val, val_probabilities
)

# The final precision/recall point has no corresponding threshold.
precision_values = precision_values[:-1]
recall_values = recall_values[:-1]

eligible_indices = np.where(recall_values >= 0.75)[0]

if len(eligible_indices) == 0:
    raise ValueError("No threshold achieved at least 75% validation recall.")

# Maximize precision while meeting the recall requirement.
best_threshold_index = eligible_indices[np.argmax(precision_values[eligible_indices])]

selected_threshold = thresholds[best_threshold_index]

validation_precision = precision_values[best_threshold_index]
validation_recall = recall_values[best_threshold_index]

print("\nSELECTED VALIDATION THRESHOLD")
print("Threshold:", round(selected_threshold, 4))
print("Validation Precision:", round(validation_precision, 4))
print("Validation Recall:", round(validation_recall, 4))


# ==========================================================
# 6. REFIT SELECTED MODEL ON TRAIN + VALIDATION
# ==========================================================

X_train_val = pd.concat([X_train, X_val])
y_train_val = pd.concat([y_train, y_val])

final_model = clone(best_model)

final_model.fit(X_train_val, y_train_val)

joblib.dump(final_model, "outputs/final_classification_model.joblib")


# ==========================================================
# 7. FINAL TEST EVALUATION - ONLY ONCE
# ==========================================================

test_probabilities = final_model.predict_proba(X_test)[:, 1]

test_predictions = (test_probabilities >= selected_threshold).astype(int)

accuracy = accuracy_score(y_test, test_predictions)
precision = precision_score(y_test, test_predictions, zero_division=0)
recall = recall_score(y_test, test_predictions, zero_division=0)
f1 = f1_score(y_test, test_predictions, zero_division=0)
auc = roc_auc_score(y_test, test_probabilities)

cm = confusion_matrix(y_test, test_predictions)

print("\n" + "=" * 65)
print("FINAL TEST SET RESULTS")
print("=" * 65)

print("Best Model:", best_model_name)
print("Selected Threshold:", round(selected_threshold, 4))

print("\nAccuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1 Score:", round(f1, 4))
print("ROC-AUC:", round(auc, 4))

print("\nConfusion Matrix:")
print(cm)


# ==========================================================
# 8. SAVE FINAL RESULTS
# ==========================================================

final_results = pd.DataFrame(
    [
        {
            "Best Model": best_model_name,
            "Threshold": selected_threshold,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "ROC-AUC": auc,
        }
    ]
)

final_results.to_csv("outputs/final_test_results.csv", index=False)

print("\nFINAL EVALUATION COMPLETED")

print("\nSaved files:")
print("outputs/validation_model_comparison.csv")
print("outputs/final_classification_model.joblib")
print("outputs/final_test_results.csv")
