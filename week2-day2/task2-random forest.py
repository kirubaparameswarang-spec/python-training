import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report


wine = load_wine()

X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = wine.target

print("=" * 60)
print("RANDOM FOREST CLASSIFICATION")
print("=" * 60)

print("\nDataset Shape:", X.shape)
print("Number of Classes:", len(wine.target_names))
print("Classes:", wine.target_names)

output_folder = "random_forest_outputs"
os.makedirs(output_folder, exist_ok=True)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)


decision_tree = DecisionTreeClassifier(random_state=42)

decision_tree.fit(X_train, y_train)

tree_predictions = decision_tree.predict(X_test)

tree_accuracy = accuracy_score(y_test, tree_predictions)

print("\n" + "=" * 60)
print("SINGLE DECISION TREE RESULTS")
print("=" * 60)

print(f"Single Tree Test Accuracy: {tree_accuracy:.4f}")


random_forest = RandomForestClassifier(
    n_estimators=200,
    oob_score=True,
    random_state=42,
    n_jobs=-1
)

random_forest.fit(X_train, y_train)

rf_predictions = random_forest.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_predictions)

print("\n" + "=" * 60)
print("RANDOM FOREST RESULTS")
print("=" * 60)

print(f"Number of Trees: 200")
print(f"Random Forest Test Accuracy: {rf_accuracy:.4f}")
print(f"OOB Score: {random_forest.oob_score_:.4f}")

print("\n" + "=" * 60)
print("DECISION TREE VS RANDOM FOREST")
print("=" * 60)

comparison = pd.DataFrame({
    "Model": ["Decision Tree", "Random Forest"],
    "Test Accuracy": [tree_accuracy, rf_accuracy]
})

print(comparison.to_string(index=False))

comparison.to_csv(
    os.path.join(output_folder, "model_comparison.csv"),
    index=False
)


estimators = [10, 50, 100, 300]
accuracy_results = []

print("\n" + "=" * 60)
print("ACCURACY FOR DIFFERENT NUMBERS OF TREES")
print("=" * 60)

for n in estimators:

    model = RandomForestClassifier(
        n_estimators=n,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    accuracy_results.append(accuracy)

    print(f"Estimators: {n:3d} | Accuracy: {accuracy:.4f}")


estimator_results = pd.DataFrame({
    "Number of Estimators": estimators,
    "Test Accuracy": accuracy_results
})

estimator_results.to_csv(
    os.path.join(output_folder, "estimator_accuracy.csv"),
    index=False
)


plt.figure(figsize=(9, 5))

plt.plot(
    estimators,
    accuracy_results,
    marker="o",
    linewidth=2
)

plt.title("Random Forest Accuracy vs Number of Trees")
plt.xlabel("Number of Estimators")
plt.ylabel("Test Accuracy")
plt.xticks(estimators)
plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(output_folder, "estimator_accuracy.png"),
    dpi=300
)

plt.show()


feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": random_forest.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n" + "=" * 60)
print("FEATURE IMPORTANCES")
print("=" * 60)

print(feature_importance.to_string(index=False))

feature_importance.to_csv(
    os.path.join(output_folder, "feature_importances.csv"),
    index=False
)

plt.figure(figsize=(10, 7))

plt.barh(
    feature_importance["Feature"],
    feature_importance["Importance"]
)

plt.title("Random Forest Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig(
    os.path.join(output_folder, "feature_importances.png"),
    dpi=300
)

plt.show()


extra_trees = ExtraTreesClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

extra_trees.fit(X_train, y_train)

extra_predictions = extra_trees.predict(X_test)

extra_accuracy = accuracy_score(y_test, extra_predictions)

print("\n" + "=" * 60)
print("BONUS: RANDOM FOREST VS EXTRA TREES")
print("=" * 60)

print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
print(f"Extra Trees Accuracy:   {extra_accuracy:.4f}")

bonus_comparison = pd.DataFrame({
    "Model": ["Random Forest", "Extra Trees"],
    "Test Accuracy": [rf_accuracy, extra_accuracy]
})

print("\n", bonus_comparison.to_string(index=False))

bonus_comparison.to_csv(
    os.path.join(output_folder, "extra_trees_comparison.csv"),
    index=False
)


print("\n" + "=" * 60)
print("RANDOM FOREST CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_test,
        rf_predictions,
        target_names=wine.target_names
    )
)


print("\n" + "=" * 60)
print("CONCEPT EXPLANATION")
print("=" * 60)

print("""
1. RANDOM FOREST:
   Random Forest is an ensemble learning algorithm that combines
   multiple decision trees to make better predictions.

2. BAGGING (BOOTSTRAP AGGREGATION):
   Bagging creates multiple random samples from the training data
   with replacement and trains a separate decision tree on each.
   Their predictions are combined using majority voting.

3. WHY RANDOM FOREST OVERFITS LESS:
   A single decision tree can memorize training data and become
   overfitted. Random Forest combines many different trees,
   reducing variance and improving generalization.

4. OOB SCORE:
   Out-of-Bag score evaluates the model using training samples
   that were not selected in a particular tree's bootstrap sample.
   It provides an internal estimate of model performance.

5. N_ESTIMATORS:
   It represents the number of decision trees in the forest.
   More trees can improve stability, but require more computation.
""")

print("\nAll Random Forest tasks completed successfully!")

print("\nOutput files saved in:", output_folder)

print("\nGenerated Files:")
for filename in sorted(os.listdir(output_folder)):
    print("-", filename)
