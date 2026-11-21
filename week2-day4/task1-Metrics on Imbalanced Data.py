"""
Task - 1: Metrics on Imbalanced Data
Week 2 - Day 4

Requirements:
1. Create an imbalanced dataset using make_classification(weights=[0.95, 0.05]).
2. Evaluate a model that always predicts the majority class.
3. Train LogisticRegression and print its classification report and confusion matrix.
4. Explain TP, FP, FN, and TN using a fraud-detection example.
5. Compare LogisticRegression with and without class_weight="balanced".
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


# ------------------------------------------------------------
# 1. Create an imbalanced dataset
# ------------------------------------------------------------
X, y = make_classification(
    n_samples=10_000,
    n_features=10,
    n_informative=5,
    n_redundant=2,
    n_classes=2,
    weights=[0.95, 0.05],
    class_sep=1.0,
    random_state=42,
)

print("=" * 65)
print("1. DATASET CLASS DISTRIBUTION")
print("=" * 65)
class_counts = pd.Series(y).value_counts().sort_index()
print(f"Class 0 (majority): {class_counts.get(0, 0)}")
print(f"Class 1 (minority): {class_counts.get(1, 0)}")
print(f"Minority-class percentage: {(y == 1).mean() * 100:.2f}%")

# Use the same split for every model so the comparison is fair.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42,
)


# ------------------------------------------------------------
# 2. Majority-class baseline
# ------------------------------------------------------------
# This model ignores the input and always predicts the most
# frequent class in the training data.
majority_class = int(pd.Series(y_train).mode()[0])
majority_predictions = np.full(shape=len(y_test), fill_value=majority_class)

majority_accuracy = accuracy_score(y_test, majority_predictions)
majority_precision = precision_score(
    y_test, majority_predictions, pos_label=1, zero_division=0
)
majority_recall = recall_score(
    y_test, majority_predictions, pos_label=1, zero_division=0
)
majority_f1 = f1_score(
    y_test, majority_predictions, pos_label=1, zero_division=0
)

print("\n" + "=" * 65)
print("2. MAJORITY-CLASS MODEL")
print("=" * 65)
print(f"Majority class predicted every time: {majority_class}")
print(f"Accuracy : {majority_accuracy:.4f}")
print(f"Precision: {majority_precision:.4f}")
print(f"Recall   : {majority_recall:.4f}")
print(f"F1-score : {majority_f1:.4f}")


# ------------------------------------------------------------
# 3. Logistic Regression without class weighting
# ------------------------------------------------------------
logistic_model = LogisticRegression(max_iter=2000, random_state=42)
logistic_model.fit(X_train, y_train)
logistic_predictions = logistic_model.predict(X_test)

print("\n" + "=" * 65)
print("3. LOGISTIC REGRESSION (WITHOUT CLASS WEIGHTING)")
print("=" * 65)
print(classification_report(
    y_test,
    logistic_predictions,
    target_names=["Class 0 (Non-fraud)", "Class 1 (Fraud)"],
    zero_division=0,
))

cm = confusion_matrix(y_test, logistic_predictions, labels=[0, 1])
print("Confusion Matrix (rows = actual, columns = predicted):")
print(cm)


# ------------------------------------------------------------
# 4. Explain the confusion matrix using fraud detection
# ------------------------------------------------------------
tn, fp, fn, tp = cm.ravel()

print("\n" + "=" * 65)
print("4. CONFUSION MATRIX EXPLANATION (FRAUD EXAMPLE)")
print("=" * 65)
print(f"True Negative (TN): {tn} - Legitimate transactions correctly predicted as legitimate.")
print(f"False Positive (FP): {fp} - Legitimate transactions incorrectly flagged as fraud.")
print(f"False Negative (FN): {fn} - Fraudulent transactions incorrectly predicted as legitimate.")
print(f"True Positive (TP): {tp} - Fraudulent transactions correctly detected as fraud.")

print("\nConfusion matrix layout:")
print("                 Predicted Non-fraud   Predicted Fraud")
print(f"Actual Non-fraud       {tn:>6}              {fp:>6}")
print(f"Actual Fraud           {fn:>6}              {tp:>6}")


# ------------------------------------------------------------
# 5. Logistic Regression with class_weight="balanced"
# ------------------------------------------------------------
balanced_model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=42,
)
balanced_model.fit(X_train, y_train)
balanced_predictions = balanced_model.predict(X_test)

balanced_accuracy = accuracy_score(y_test, balanced_predictions)
balanced_precision = precision_score(
    y_test, balanced_predictions, pos_label=1, zero_division=0
)
balanced_recall = recall_score(
    y_test, balanced_predictions, pos_label=1, zero_division=0
)
balanced_f1 = f1_score(
    y_test, balanced_predictions, pos_label=1, zero_division=0
)

print("\n" + "=" * 65)
print('5. LOGISTIC REGRESSION (class_weight="balanced")')
print("=" * 65)
print(classification_report(
    y_test,
    balanced_predictions,
    target_names=["Class 0 (Non-fraud)", "Class 1 (Fraud)"],
    zero_division=0,
))
balanced_cm = confusion_matrix(y_test, balanced_predictions, labels=[0, 1])
print("Confusion Matrix (rows = actual, columns = predicted):")
print(balanced_cm)


# ------------------------------------------------------------
# 6. Compare the models
# ------------------------------------------------------------
comparison = pd.DataFrame(
    {
        "Model": [
            "Majority-class baseline",
            "Logistic Regression",
            'Logistic Regression (class_weight="balanced")',
        ],
        "Accuracy": [
            majority_accuracy,
            accuracy_score(y_test, logistic_predictions),
            balanced_accuracy,
        ],
        "Precision (fraud)": [
            majority_precision,
            precision_score(y_test, logistic_predictions, zero_division=0),
            balanced_precision,
        ],
        "Recall (fraud)": [
            majority_recall,
            recall_score(y_test, logistic_predictions, zero_division=0),
            balanced_recall,
        ],
        "F1 (fraud)": [
            majority_f1,
            f1_score(y_test, logistic_predictions, zero_division=0),
            balanced_f1,
        ],
    }
)

print("\n" + "=" * 65)
print("6. MODEL COMPARISON")
print("=" * 65)
print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

print("\n" + "=" * 65)
print("7. CONCEPTS")
print("=" * 65)
print(
    "Accuracy: The proportion of all predictions that are correct.\n"
    "Precision: Of the transactions predicted as fraud, the proportion that really are fraud.\n"
    "Recall: Of all actual fraudulent transactions, the proportion detected by the model.\n"
    "F1-score: The harmonic mean of precision and recall.\n"
    "Class imbalance: One class has many more samples than the other. A high accuracy\n"
    "can be misleading when the model misses most examples of the minority class.\n"
    'class_weight="balanced" gives more importance to the minority class during training.\n'
    "It often improves fraud recall (and sometimes F1), but may reduce precision or accuracy.\n"
    "The exact change depends on the dataset and should be checked using the printed metrics."
)
