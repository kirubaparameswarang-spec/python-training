"""
Task 2 - ROC Curve and AUC
Week 2 - Day 4

This script uses the same imbalanced dataset setup as Task 1:
10,000 samples, 10 features, and approximately 95% class 0 / 5% class 1.

Models:
1. Logistic Regression (with class_weight="balanced")
2. Random Forest
3. XGBoost

Install dependencies if needed:
    pip install numpy pandas matplotlib scikit-learn xgboost
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except ImportError as error:
    raise ImportError(
        "XGBoost is not installed. Run: pip install xgboost"
    ) from error


# ---------------------------------------------------------
# 1. Create the same imbalanced dataset used in Task 1
# ---------------------------------------------------------

X, y = make_classification(
    n_samples=10000,
    n_features=10,
    n_informative=5,
    n_redundant=2,
    n_classes=2,
    weights=[0.95, 0.05],
    class_sep=1.0,
    random_state=42,
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print("=" * 65)
print("TASK 2 - ROC CURVE AND AUC")
print("=" * 65)
print(f"Training samples: {len(X_train)}")
print(f"Testing samples : {len(X_test)}")
print(f"Class 0 in test : {(y_test == 0).sum()}")
print(f"Class 1 in test : {(y_test == 1).sum()}")


# ---------------------------------------------------------
# 2. Train the three models
# ---------------------------------------------------------

positive_count = int((y_train == 1).sum())
negative_count = int((y_train == 0).sum())
scale_pos_weight = negative_count / positive_count

models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=42,
        ),
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    ),
    "XGBoost": XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    ),
}

# Store predicted probabilities for the positive class (class 1).
probabilities = {}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    probabilities[model_name] = model.predict_proba(X_test)[:, 1]
    print(f"\n{model_name} training completed.")


# ---------------------------------------------------------
# 3. Bonus function: highest precision at minimum recall
# ---------------------------------------------------------

def best_threshold(y_true, proba, min_recall):
    """
    Return the threshold with the highest precision while recall
    is at least min_recall.

    Returns:
        (best_threshold, precision, recall)

    Raises:
        ValueError if no available threshold meets min_recall.
    """
    if not 0 <= min_recall <= 1:
        raise ValueError("min_recall must be between 0 and 1.")

    precision_values, recall_values, thresholds = precision_recall_curve(
        y_true, proba
    )

    # precision and recall have one more element than thresholds.
    valid_indices = np.where(recall_values[:-1] >= min_recall)[0]

    if len(valid_indices) == 0:
        raise ValueError(
            f"No threshold found that achieves recall >= {min_recall:.2f}."
        )

    # Among thresholds meeting the recall requirement, select
    # the one with the highest precision.
    best_index = valid_indices[
        np.argmax(precision_values[valid_indices])
    ]

    return (
        float(thresholds[best_index]),
        float(precision_values[best_index]),
        float(recall_values[best_index]),
    )


# ---------------------------------------------------------
# 4. Plot all three ROC curves and calculate AUC
# ---------------------------------------------------------

output_dir = Path(__file__).resolve().parent / "task2_roc_outputs"
output_dir.mkdir(exist_ok=True)

plt.figure(figsize=(9, 7))

results = []

print("\n" + "=" * 65)
print("AUC, THRESHOLD, PRECISION AND RECALL")
print("=" * 65)

for model_name, proba in probabilities.items():
    fpr, tpr, _ = roc_curve(y_test, proba)
    model_auc = roc_auc_score(y_test, proba)

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"{model_name} (AUC = {model_auc:.4f})",
    )

    threshold, threshold_precision, threshold_recall = best_threshold(
        y_test, proba, min_recall=0.80
    )

    predictions_at_threshold = (proba >= threshold).astype(int)
    model_f1 = f1_score(y_test, predictions_at_threshold, zero_division=0)

    results.append(
        {
            "Model": model_name,
            "AUC": model_auc,
            "F1": model_f1,
        }
    )

    print(f"\n{model_name}")
    print(f"AUC                  : {model_auc:.4f}")
    print(f"Selected threshold   : {threshold:.6f}")
    print(f"Precision at threshold: {threshold_precision:.4f}")
    print(f"Recall at threshold  : {threshold_recall:.4f}")
    print(f"F1 at selected threshold: {model_f1:.4f}")

plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curves - Imbalanced Classification")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
roc_path = output_dir / "roc_curves.png"
plt.savefig(roc_path, dpi=300)
plt.show()


# ---------------------------------------------------------
# 5. Print the required Model | AUC | F1 comparison table
# ---------------------------------------------------------

results_df = pd.DataFrame(results)
results_df["AUC"] = results_df["AUC"].round(4)
results_df["F1"] = results_df["F1"].round(4)

print("\n" + "=" * 65)
print("MODEL COMPARISON TABLE")
print("=" * 65)
print(results_df.to_string(index=False))

csv_path = output_dir / "model_comparison.csv"
results_df.to_csv(csv_path, index=False)


# ---------------------------------------------------------
# 6. Plot the Precision-Recall curves
# ---------------------------------------------------------

plt.figure(figsize=(9, 7))

for model_name, proba in probabilities.items():
    precision_values, recall_values, _ = precision_recall_curve(
        y_test, proba
    )
    pr_auc = auc(recall_values, precision_values)

    plt.plot(
        recall_values,
        precision_values,
        linewidth=2,
        label=f"{model_name} (PR AUC = {pr_auc:.4f})",
    )

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curves - Imbalanced Classification")
plt.legend(loc="best")
plt.grid(alpha=0.3)
plt.tight_layout()
pr_path = output_dir / "precision_recall_curves.png"
plt.savefig(pr_path, dpi=300)
plt.show()


# ---------------------------------------------------------
# 7. Explain F1 vs AUC and the better curve for imbalanced data
# ---------------------------------------------------------

print("\n" + "=" * 65)
print("F1 VS AUC - WHEN TO USE")
print("=" * 65)
print("1. Use F1 when you need a balance between precision and recall.")
print("2. F1 is useful when false positives and false negatives both matter.")
print("3. Use ROC-AUC to compare how well models rank positive cases above negative cases.")
print("4. ROC-AUC evaluates ranking across thresholds, not just one threshold.")

print("\nPRECISION-RECALL VS ROC:")
print(
    "For highly imbalanced data, the Precision-Recall curve is usually "
    "more informative because it focuses on the minority positive class "
    "and shows the trade-off between precision and recall."
)

print("\nSaved output files:")
print(f"ROC chart              : {roc_path}")
print(f"Precision-Recall chart : {pr_path}")
print(f"Model comparison CSV   : {csv_path}")
print("\nTask 2 completed successfully.")
