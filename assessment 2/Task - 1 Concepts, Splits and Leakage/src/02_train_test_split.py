
import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("data/student_performance.csv")

print("STUDENT PERFORMANCE DATASET")
print(df)

# Create binary target: Pass = 1, Fail = 0
df["Pass_Fail"] = (df["Final_Score"] >= 65).astype(int)

# Separate features and target
# Final_Score is excluded to prevent target leakage
X = df[["Hours_Studied", "Attendance", "Previous_Score"]]
y = df["Pass_Fail"]

# First split: 80% development data and 20% test data
X_temp, X_test, y_temp, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Second split: 60% training and 20% validation
X_train, X_val, y_train, y_val = train_test_split(
    X_temp,
    y_temp,
    test_size=0.25,
    random_state=42,
    stratify=y_temp
)

# Display dataset split summary
print("\nDATASET SPLIT SUMMARY")
print("Original records:", len(df))
print("Training records:", len(X_train))
print("Validation records:", len(X_val))
print("Testing records:", len(X_test))

# Display class distributions
print("\nCLASS DISTRIBUTION")
print("Training:\n", y_train.value_counts())
print("Validation:\n", y_val.value_counts())
print("Testing:\n", y_test.value_counts())

# Verify split sizes
assert len(X_train) == 12
assert len(X_val) == 4
assert len(X_test) == 4

print("\n60/20/20 STRATIFIED SPLIT COMPLETED SUCCESSFULLY")

# Keep X_test and y_test aside for final evaluation only.
