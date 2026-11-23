
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Load dataset
df = pd.read_csv("data/student_performance.csv")

# Create binary target
df["Pass_Fail"] = (df["Final_Score"] >= 65).astype(int)

# Features and target
X = df[["Hours_Studied", "Attendance", "Previous_Score"]]
y = df["Pass_Fail"]

# 60/20/20 stratified split
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=0.25,
    random_state=42,
    stratify=y_temp
)

# Decision Tree depths
depths = [1, 3, 5, 10, None]

results = []

for depth in depths:

    model = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )

    model.fit(X_train, y_train)

    train_accuracy = model.score(X_train, y_train)
    val_accuracy = model.score(X_val, y_val)

    results.append({
        "max_depth": str(depth),
        "training_accuracy": train_accuracy,
        "validation_accuracy": val_accuracy
    })

# Display results
results_df = pd.DataFrame(results)

print("\nDECISION TREE OVERFITTING EXPERIMENT")
print(results_df)

# Plot accuracy comparison
plt.figure(figsize=(9, 5))

plt.plot(
    results_df["max_depth"],
    results_df["training_accuracy"],
    marker="o",
    label="Training Accuracy"
)

plt.plot(
    results_df["max_depth"],
    results_df["validation_accuracy"],
    marker="o",
    label="Validation Accuracy"
)

plt.xlabel("Decision Tree Maximum Depth")
plt.ylabel("Accuracy")
plt.title("Decision Tree Overfitting Experiment")
plt.legend()
plt.grid(True)

plt.savefig("overfitting_experiment.png")
plt.show()

print("\nGraph saved as overfitting_experiment.png")
