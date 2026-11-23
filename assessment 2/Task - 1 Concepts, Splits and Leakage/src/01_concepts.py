
"""
Assessment 2: Concepts, Splits and Leakage

Topic 1: Understanding Machine Learning Concepts
"""

from pathlib import Path

import pandas as pd

# Load the shared student performance dataset
DATA_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "student_performance.csv"
)

df = pd.read_csv(DATA_PATH)

# Display dataset
print("STUDENT PERFORMANCE DATASET")
print(df)

# Features (Independent Variables)
X = df[["Hours_Studied", "Attendance", "Previous_Score"]]

# Target (Dependent Variable)
y = df["Final_Score"]

print("\nFEATURES:")
print(X)

print("\nTARGET:")
print(y)

# Basic dataset information
print("\nDATASET INFORMATION:")
df.info()

print("\nSTATISTICAL SUMMARY:")
print(df.describe())

# Check missing values
print("\nMISSING VALUES:")
print(df.isnull().sum())

# Display feature and target shapes
print("\nFEATURE SHAPE:", X.shape)
print("TARGET SHAPE:", y.shape)
