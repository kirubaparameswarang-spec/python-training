"""
Task 1: Decision Tree Classification
Dataset: scikit-learn Wine dataset

Run from the project .venv:
    python task1_decision_tree.py

Outputs are saved in:
    week2-day2/decision_tree_outputs/
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree


# Keep all generated submission files beside this script.
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "decision_tree_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# 1. Load the Wine dataset and split it into training and testing data.
wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = wine.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("=" * 65)
print("TASK 1: DECISION TREE CLASSIFICATION")
print("=" * 65)
print(f"Training samples: {len(X_train)}")
print(f"Testing samples : {len(X_test)}")


# 2. Train and compare Gini and Entropy decision trees.
print("\n1. COMPARISON: GINI VS ENTROPY")

criterion_results = []

for criterion in ["gini", "entropy"]:
    model = DecisionTreeClassifier(
        criterion=criterion,
        random_state=42,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    criterion_results.append(
        {"Criterion": criterion, "Test Accuracy": accuracy}
    )
    print(f"{criterion.capitalize():8s} test accuracy: {accuracy:.4f}")

criterion_df = pd.DataFrame(criterion_results)
criterion_df.to_csv(OUTPUT_DIR / "criterion_comparison.csv", index=False)


# 3. Tune max_depth and min_samples_leaf using cross-validation.
print("\n2. HYPERPARAMETER TUNING")

parameter_grid = {
    "max_depth": [2, 3, 4, 5, 6, None],
    "min_samples_leaf": [1, 2, 3, 4, 5],
}

grid_search = GridSearchCV(
    estimator=DecisionTreeClassifier(
        criterion="gini",
        random_state=42,
    ),
    param_grid=parameter_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
best_predictions = best_model.predict(X_test)
best_test_accuracy = accuracy_score(y_test, best_predictions)

print(f"Best combination: {grid_search.best_params_}")
print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")
print(f"Best model test accuracy: {best_test_accuracy:.4f}")

tuning_df = pd.DataFrame(grid_search.cv_results_)[
    [
        "param_max_depth",
        "param_min_samples_leaf",
        "mean_test_score",
        "rank_test_score",
    ]
].sort_values(["rank_test_score", "mean_test_score"], ascending=[True, False])

tuning_df.to_csv(OUTPUT_DIR / "hyperparameter_tuning_results.csv", index=False)


# 4. Plot the tuned decision tree, showing at most 3 levels of depth.
print("\n3. DECISION TREE PLOT")

plt.figure(figsize=(22, 12))
plot_tree(
    best_model,
    feature_names=wine.feature_names,
    class_names=wine.target_names,
    filled=True,
    rounded=True,
    max_depth=3,
    fontsize=8,
)
plt.title("Wine Decision Tree (display limited to max_depth=3)")
plt.tight_layout()

tree_image_path = OUTPUT_DIR / "decision_tree.png"
plt.savefig(tree_image_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Tree image saved to: {tree_image_path}")


# 5. Explain the root split and the next split on its left branch.
def describe_split(tree_model, node_id, split_label):
    tree = tree_model.tree_

    if tree.feature[node_id] < 0:
        print(f"{split_label}: this node is a leaf; no further split.")
        return

    feature_name = wine.feature_names[tree.feature[node_id]]
    threshold = tree.threshold[node_id]

    print(
        f"{split_label}: The tree checks whether '{feature_name}' "
        f"<= {threshold:.3f}."
    )
    print(
        "  Left branch: the condition is true (value is at or below "
        "the threshold)."
    )
    print(
        "  Right branch: the condition is false (value is above "
        "the threshold)."
    )


print("\n4. FIRST TWO SPLITS IN PLAIN WORDS")
tree_structure = best_model.tree_
describe_split(best_model, 0, "First split (root)")

left_child = tree_structure.children_left[0]
if left_child >= 0:
    describe_split(best_model, left_child, "Second split (left branch)")
else:
    print("Second split: the root is already a leaf for this fitted tree.")


# 6. Print and save the five most important features.
print("\n5. TOP 5 FEATURE IMPORTANCES")

importance_df = pd.DataFrame(
    {
        "Feature": wine.feature_names,
        "Importance": best_model.feature_importances_,
    }
).sort_values("Importance", ascending=False)

top_five = importance_df.head(5).reset_index(drop=True)
top_five.insert(0, "Rank", range(1, len(top_five) + 1))

print(top_five.to_string(index=False))

importance_csv_path = OUTPUT_DIR / "feature_importance_top5.csv"
top_five.to_csv(importance_csv_path, index=False)


# 7. Gini impurity explanation.
print("\n6. GINI IMPURITY (3 LINES)")
print("1. Gini impurity measures how mixed the classes are in a tree node.")
print("2. A value of 0 means the node contains samples from only one class.")
print("3. The tree prefers splits that reduce impurity and create purer groups.")

print("\nAll task outputs have been saved in:")
print(OUTPUT_DIR)
print("\nSubmission files:")
print("- decision_tree.png")
print("- feature_importance_top5.csv")
print("- criterion_comparison.csv")
print("- hyperparameter_tuning_results.csv")
