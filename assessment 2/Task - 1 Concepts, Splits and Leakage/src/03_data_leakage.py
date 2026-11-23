import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Locate dataset
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "student_performance.csv"

# Load dataset
df = pd.read_csv(DATA_PATH)

print("STUDENT PERFORMANCE DATASET")
print(df)

# Separate features and target
X = df.drop("Final_Score", axis=1)
y = df["Final_Score"]

# Split dataset first
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ------------------------------------------------
# INCORRECT APPROACH: DATA LEAKAGE
# ------------------------------------------------

print("\nINCORRECT APPROACH - DATA LEAKAGE")

scaler_wrong = StandardScaler()

# Incorrect: fit scaler using all records
X_scaled_wrong = scaler_wrong.fit_transform(X)

# Preserve original row indices
X_scaled_wrong = pd.DataFrame(
    X_scaled_wrong,
    columns=X.columns,
    index=X.index
)

print("\nIncorrectly scaled training data:")
print(X_scaled_wrong.loc[X_train.index])

print("\nIncorrectly scaled testing data:")
print(X_scaled_wrong.loc[X_test.index])
# ------------------------------------------------
# CORRECT APPROACH: PREVENT DATA LEAKAGE
# ------------------------------------------------

print("\nCORRECT APPROACH - NO DATA LEAKAGE")

scaler = StandardScaler()

# Fit only on training features
X_train_scaled = scaler.fit_transform(X_train)

# Transform testing features using training statistics
X_test_scaled = scaler.transform(X_test)

print("\nTRAINING FEATURES AFTER SCALING:")
print(X_train_scaled)

print("\nTESTING FEATURES AFTER SCALING:")
print(X_test_scaled)

print("\nDATA LEAKAGE PREVENTION SUMMARY")
print("Training shape:", X_train_scaled.shape)
print("Testing shape:", X_test_scaled.shape)
print("Scaler fitted only on training data.")
print("Testing data was not used during fitting.")
