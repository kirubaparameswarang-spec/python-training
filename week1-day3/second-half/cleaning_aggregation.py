
import pandas as pd
import os

# Load orders.csv from the same folder
file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'orders.csv')
df = pd.read_csv(file_path)

# Find missing values
missing = df.isna().sum()

# Find duplicates
duplicates = df.duplicated().sum()

print("Missing: amount", missing['amount'])
print("Duplicates:", duplicates)

# Remove duplicate rows
df = df.drop_duplicates()

# Calculate median and fill missing values
median_value = df['amount'].median()
df['amount'] = df['amount'].fillna(median_value)

print("Median used:", median_value)

# Convert data types
df['amount'] = pd.to_numeric(df['amount'])
df['order_id'] = df['order_id'].astype(str)

# Group and aggregate
result = df.groupby('customer_id').agg(
    total=('amount', 'sum'),
    orders=('order_id', 'count')
)

# Print final result
print(result.to_string())