import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve


# -----------------------------------------
# 1. LOAD DATA AND SAVED MODELS
# -----------------------------------------

df = pd.read_csv("data/loan_preprocessing.csv")

models = joblib.load("outputs/classification_models.joblib")

splits = np.load("outputs/data_splits.npz")

val_idx = splits["val_idx"]

X = df.drop(columns=["loan_status", "application_date"])

y = df["loan_status"]

X_val = X.iloc[val_idx]
y_val = y.iloc[val_idx]

os.makedirs("outputs", exist_ok=True)


# -----------------------------------------
# 2. SELECT THREE CLASSIFICATION MODELS
# -----------------------------------------

selected_models = ["Logistic Regression", "Random Forest", "XGBoost"]

print("=" * 55)
print("ROC CURVE AND THRESHOLD EVALUATION")
print("=" * 55)


# -----------------------------------------
# 3. PLOT ROC CURVES
# -----------------------------------------

plt.figure(figsize=(9, 7))

roc_results = []

for name in selected_models:
    model = models[name]

    probabilities = model.predict_proba(X_val)[:, 1]

    fpr, tpr, thresholds = roc_curve(y_val, probabilities)

    auc_score = roc_auc_score(y_val, probabilities)

    plt.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC = {auc_score:.4f})")

    roc_results.append({"Model": name, "ROC_AUC": auc_score})

    print(f"{name} ROC-AUC: {auc_score:.4f}")


# Random classifier reference line
plt.plot([0, 1], [0, 1], linestyle="--", label="Random Classifier")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison - Three Models")

plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig("outputs/roc_curves.png", dpi=300)

plt.close()

print("\nROC chart saved:")
print("outputs/roc_curves.png")


# -----------------------------------------
# 4. SELECT BEST THRESHOLD FOR 75% RECALL
# -----------------------------------------

# XGBoost is selected for threshold analysis
# because it achieved the highest validation F1.

best_model_name = "XGBoost"

best_model = models[best_model_name]

probabilities = best_model.predict_proba(X_val)[:, 1]

precision_values, recall_values, thresholds = precision_recall_curve(
    y_val, probabilities
)

# precision and recall have one more element
# than thresholds, so use [:-1].

valid_indices = np.where(recall_values[:-1] >= 0.75)[0]

if len(valid_indices) == 0:
    raise ValueError("No threshold achieved the required 75% recall.")

# Choose the qualifying threshold with the highest precision.
# If tied, choose the highest threshold.

best_precision = precision_values[valid_indices].max()

best_indices = valid_indices[precision_values[valid_indices] == best_precision]

best_index = best_indices[np.argmax(thresholds[best_indices])]

selected_threshold = thresholds[best_index]

selected_precision = precision_values[best_index]

selected_recall = recall_values[best_index]


# -----------------------------------------
# 5. PRINT THRESHOLD RESULTS
# -----------------------------------------

print("\nDECISION THRESHOLD EVALUATION")

print("Model:", best_model_name)

print("Required Recall: 75%")

print(f"Selected Threshold: {selected_threshold:.4f}")

print(f"Recall: {selected_recall:.4f}")

print(f"Precision: {selected_precision:.4f}")


# -----------------------------------------
# 6. SAVE ROC RESULTS
# -----------------------------------------

pd.DataFrame(roc_results).to_csv("outputs/roc_results.csv", index=False)


# -----------------------------------------
# 7. SAVE THRESHOLD RESULTS
# -----------------------------------------

threshold_results = pd.DataFrame(
    [
        {
            "Model": best_model_name,
            "Required_Recall": 0.75,
            "Selected_Threshold": selected_threshold,
            "Recall": selected_recall,
            "Precision": selected_precision,
        }
    ]
)

threshold_results.to_csv("outputs/threshold_results.csv", index=False)

print("\nOUTPUT FILES SAVED")
print("outputs/roc_curves.png")
print("outputs/roc_results.csv")
print("outputs/threshold_results.csv")

print("\nROC AND THRESHOLD EVALUATION COMPLETED")
