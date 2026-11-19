"""
Task - 2: Scaling
Folder: week2-day3
File: task1-Scaling.py

Requirements:
- StandardScaler on age and income; verify mean ~0 and standard deviation ~1.
- MinMaxScaler; verify values are in the range 0 to 1.
- Compare KNN and Decision Tree accuracy with and without scaling.
- Fit scalers on training data only to prevent data leakage.
- Bonus: compare RobustScaler when outliers are present.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


# ---------------------------------------------------------------------
# 1. StandardScaler and MinMaxScaler on age and income
# ---------------------------------------------------------------------
print("=" * 65)
print("1. SCALING AGE AND INCOME")
print("=" * 65)

# Small example dataset to demonstrate scaling age and income.
people = pd.DataFrame({
    "age": [22, 25, 29, 35, 42, 48, 55, 61],
    "income": [18000, 22000, 28000, 35000, 48000, 62000, 85000, 110000]
})

# Split before fitting either scaler. Only training rows are used by fit().
age_income_train, age_income_test = train_test_split(
    people, test_size=0.25, random_state=42
)

standard_scaler = StandardScaler()
age_income_train_standard = standard_scaler.fit_transform(
    age_income_train[["age", "income"]]
)
age_income_test_standard = standard_scaler.transform(
    age_income_test[["age", "income"]]
)
standard_train_df = pd.DataFrame(
    age_income_train_standard, columns=["age", "income"]
)

print("\nOriginal age and income:")
print(people)
print("\nStandardScaler - scaled training data:")
print(standard_train_df.round(3))
print("\nTraining means (approximately 0):")
print(standard_train_df.mean().round(6))
print("Training standard deviations (approximately 1):")
print(standard_train_df.std(ddof=0).round(6))

minmax_scaler = MinMaxScaler()
age_income_train_minmax = minmax_scaler.fit_transform(
    age_income_train[["age", "income"]]
)
age_income_test_minmax = minmax_scaler.transform(
    age_income_test[["age", "income"]]
)
minmax_train_df = pd.DataFrame(
    age_income_train_minmax, columns=["age", "income"]
)

print("\nMinMaxScaler - scaled training data:")
print(minmax_train_df.round(3))
print("Training minimum values (should be 0):")
print(minmax_train_df.min().round(6))
print("Training maximum values (should be 1):")
print(minmax_train_df.max().round(6))
print("Note: transformed test values can fall outside 0 to 1 if they")
print("are beyond the minimum or maximum values seen in training.")


# ---------------------------------------------------------------------
# 2. Load breast cancer dataset and split before scaling
# ---------------------------------------------------------------------
print("\n" + "=" * 65)
print("2. KNN AND DECISION TREE: WITH VS WITHOUT SCALING")
print("=" * 65)

cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

# Split first. The test set must not be used to fit the scaler.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# StandardScaler is fitted ONLY on training data.
cancer_scaler = StandardScaler()
X_train_scaled = cancer_scaler.fit_transform(X_train)
X_test_scaled = cancer_scaler.transform(X_test)


# ---------------------------------------------------------------------
# 3. K-Nearest Neighbors (KNN)
# ---------------------------------------------------------------------
knn_unscaled = KNeighborsClassifier(n_neighbors=5)
knn_unscaled.fit(X_train, y_train)
knn_accuracy_unscaled = knn_unscaled.score(X_test, y_test)

knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_scaled, y_train)
knn_accuracy_scaled = knn_scaled.score(X_test_scaled, y_test)

print("\nKNN accuracy:")
print(f"KNN without scaling: {knn_accuracy_unscaled:.2f}")
print(f"KNN with scaling:    {knn_accuracy_scaled:.2f}")


# ---------------------------------------------------------------------
# 4. Decision Tree
# ---------------------------------------------------------------------
tree_unscaled = DecisionTreeClassifier(random_state=42)
tree_unscaled.fit(X_train, y_train)
tree_accuracy_unscaled = tree_unscaled.score(X_test, y_test)

tree_scaled = DecisionTreeClassifier(random_state=42)
tree_scaled.fit(X_train_scaled, y_train)
tree_accuracy_scaled = tree_scaled.score(X_test_scaled, y_test)

print("\nDecision Tree accuracy:")
print(f"Decision Tree without scaling: {tree_accuracy_unscaled:.2f}")
print(f"Decision Tree with scaling:    {tree_accuracy_scaled:.2f}")


# ---------------------------------------------------------------------
# 5. Bonus: RobustScaler with outliers
# ---------------------------------------------------------------------
print("\n" + "=" * 65)
print("3. BONUS: ROBUSTSCALER WITH OUTLIERS")
print("=" * 65)

outlier_data = pd.DataFrame({
    "age": [21, 23, 25, 27, 29, 31, 33, 35, 90],
    "income": [20000, 22000, 24000, 26000, 28000, 30000, 32000, 34000, 500000]
})

outlier_standard = StandardScaler().fit_transform(outlier_data)
outlier_robust = RobustScaler().fit_transform(outlier_data)

print("\nOriginal data containing outliers:")
print(outlier_data)
print("\nStandardScaler result:")
print(pd.DataFrame(outlier_standard, columns=["age", "income"]).round(2))
print("\nRobustScaler result:")
print(pd.DataFrame(outlier_robust, columns=["age", "income"]).round(2))


# ---------------------------------------------------------------------
# 6. Accuracy comparison chart
# ---------------------------------------------------------------------
labels = ["KNN", "Decision Tree"]
without_scaling = [knn_accuracy_unscaled, tree_accuracy_unscaled]
with_scaling = [knn_accuracy_scaled, tree_accuracy_scaled]

x = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(8, 5))
plt.bar(x - width / 2, without_scaling, width, label="Without scaling")
plt.bar(x + width / 2, with_scaling, width, label="With scaling")
plt.xticks(x, labels)
plt.ylabel("Test accuracy")
plt.ylim(0, 1.05)
plt.title("Model Accuracy: With vs Without Scaling")
plt.legend()
plt.tight_layout()
plt.savefig("scaling_accuracy_comparison.png", dpi=300)
plt.show()


# ---------------------------------------------------------------------
# 7. Concepts and conclusion
# ---------------------------------------------------------------------
print("\n" + "=" * 65)
print("4. CONCEPTS AND CONCLUSION")
print("=" * 65)

print("""
StandardScaler:
- Centers each feature around mean 0 and scales it to standard deviation 1.
- Useful when features have different units or ranges.

MinMaxScaler:
- Rescales each feature to a specified range; by default, 0 to 1.
- Can be strongly affected by extreme outliers.

Why scaling matters for KNN:
- KNN uses distances between data points.
- A feature with large numeric values can dominate distance calculations.
- Scaling gives features a more comparable contribution.

Why scaling usually does not change a Decision Tree:
- Decision Trees split using thresholds on individual features.
- Scaling changes the numeric threshold but generally preserves the ordering
  of values, so the same partitions can be made.

Why fit the scaler only on training data:
- Fitting on the full dataset allows information from the test set to leak
  into training.
- Correct approach: fit_transform(X_train), then transform(X_test).

RobustScaler:
- Uses the median and interquartile range.
- It is less affected by outliers than StandardScaler.
""")

print("Done. Accuracy chart saved as scaling_accuracy_comparison.png")
