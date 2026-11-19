# Task 2 - K-Means Clustering
# Week 2 - Day 2
#
# Requirements:
# 1. Create 300 points with make_blobs(centers=4, random_state=7).
# 2. Run K-Means for k = 2 to 8 and plot the elbow curve (inertia).
# 3. Calculate silhouette scores and choose the best k.
# 4. Plot final clusters with centroids.
# 5. Apply K-Means to Mall Customers-style data and describe each segment.
# Bonus: Tune XGBoost max_depth and n_estimators using GridSearchCV.

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs, load_diabetes
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV


# ============================================================
# PART 1: Create 300 points and find the best K
# ============================================================

print("=" * 60)
print("PART 1: K-MEANS CLUSTERING WITH MAKE_BLOBS")
print("=" * 60)

X, y_true = make_blobs(
    n_samples=300,
    centers=4,
    random_state=7
)

k_values = range(2, 9)
inertias = []
silhouette_scores = []

for k in k_values:
    kmeans = KMeans(
        n_clusters=k,
        random_state=7,
        n_init=10
    )
    labels = kmeans.fit_predict(X)

    inertias.append(kmeans.inertia_)
    score = silhouette_score(X, labels)
    silhouette_scores.append(score)

    print(f"k = {k}: Inertia = {kmeans.inertia_:.2f}, "
          f"Silhouette Score = {score:.3f}")

best_k = list(k_values)[int(np.argmax(silhouette_scores))]
best_score = max(silhouette_scores)

print(f"\nBest k by silhouette: {best_k} "
      f"(score = {best_score:.3f})")

# Elbow curve
plt.figure(figsize=(8, 5))
plt.plot(list(k_values), inertias, marker="o")
plt.title("Elbow Method")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.xticks(list(k_values))
plt.grid(True)
plt.tight_layout()
plt.show()

# Silhouette score plot
plt.figure(figsize=(8, 5))
plt.plot(list(k_values), silhouette_scores, marker="o")
plt.title("Silhouette Score for Different K Values")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette Score")
plt.xticks(list(k_values))
plt.grid(True)
plt.tight_layout()
plt.show()


# ============================================================
# PART 2: Final clusters and centroids
# ============================================================

print("\n" + "=" * 60)
print("PART 2: FINAL CLUSTERS")
print("=" * 60)

final_kmeans = KMeans(
    n_clusters=best_k,
    random_state=7,
    n_init=10
)
final_labels = final_kmeans.fit_predict(X)
centroids = final_kmeans.cluster_centers_

plt.figure(figsize=(8, 6))
plt.scatter(
    X[:, 0], X[:, 1],
    c=final_labels,
    cmap="viridis",
    s=35,
    label="Data points"
)
plt.scatter(
    centroids[:, 0], centroids[:, 1],
    c="red",
    marker="X",
    s=250,
    edgecolor="black",
    label="Centroids"
)
plt.title(f"Final K-Means Clusters (k={best_k})")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# ============================================================
# PART 3: Mall Customers-style segmentation
# ============================================================

print("\n" + "=" * 60)
print("PART 3: MALL CUSTOMERS SEGMENTATION")
print("=" * 60)

# If Mall_Customers.csv is in the same folder as this script,
# the script uses it. Otherwise, it creates sample Mall-style data.
csv_path = "Mall_Customers.csv"

try:
    customers = pd.read_csv(csv_path)

    # Support common column-name variations.
    normalized_columns = {
        col.strip().lower().replace("_", " ").replace("-", " "): col
        for col in customers.columns
    }

    income_col = next(
        (original for normalized, original in normalized_columns.items()
         if "annual income" in normalized),
        None
    )
    spending_col = next(
        (original for normalized, original in normalized_columns.items()
         if "spending score" in normalized),
        None
    )

    if income_col is None or spending_col is None:
        raise ValueError(
            "CSV must contain Annual Income and Spending Score columns."
        )

    mall_data = customers[[income_col, spending_col]].dropna().copy()
    mall_data.columns = ["Annual Income (k$)", "Spending Score (1-100)"]
    print(f"Loaded customer data from {csv_path}")

except FileNotFoundError:
    print("Mall_Customers.csv not found. Creating sample Mall-style data.")

    rng = np.random.default_rng(7)
    sample_centers = np.array([
        [25, 25],   # Low income, low spending
        [25, 75],   # Low income, high spending
        [75, 25],   # High income, low spending
        [75, 75],   # High income, high spending
        [50, 50]    # Average income, average spending
    ])

    sample_sizes = [40, 40, 40, 40, 40]
    sample_rows = []

    for center, size in zip(sample_centers, sample_sizes):
        points = rng.normal(loc=center, scale=[8, 10], size=(size, 2))
        points[:, 0] = np.clip(points[:, 0], 15, 140)
        points[:, 1] = np.clip(points[:, 1], 1, 100)
        sample_rows.append(points)

    mall_data = pd.DataFrame(
        np.vstack(sample_rows),
        columns=["Annual Income (k$)", "Spending Score (1-100)"]
    )

# Scale the two features before clustering.
scaler = StandardScaler()
mall_scaled = scaler.fit_transform(mall_data)

# Five clusters are commonly used for this Mall Customers-style example.
mall_kmeans = KMeans(
    n_clusters=5,
    random_state=7,
    n_init=10
)
mall_data["Cluster"] = mall_kmeans.fit_predict(mall_scaled)

# Convert centroids back to the original income/spending scale.
mall_centers = scaler.inverse_transform(mall_kmeans.cluster_centers_)

print("\nCustomer segment averages:")
segment_summary = (
    mall_data.groupby("Cluster")[
        ["Annual Income (k$)", "Spending Score (1-100)"]
    ].mean().round(2)
)
print(segment_summary)

print("\nSegment descriptions:")
for cluster_id, row in segment_summary.iterrows():
    income = row["Annual Income (k$)"]
    spending = row["Spending Score (1-100)"]

    if income >= 60 and spending >= 60:
        description = "High-income, high-spending customers; valuable premium target group."
    elif income >= 60 and spending < 40:
        description = "High-income, low-spending customers; potential to increase engagement."
    elif income < 40 and spending >= 60:
        description = "Low-income, high-spending customers; frequent or enthusiastic shoppers."
    elif income < 40 and spending < 40:
        description = "Low-income, low-spending customers; budget-conscious group."
    else:
        description = "Average-income, average-spending customers; suitable for general promotions."

    print(f"Cluster {cluster_id}: {description}")

# Plot customer segments and their centroids.
plt.figure(figsize=(9, 6))
scatter = plt.scatter(
    mall_data["Annual Income (k$)"],
    mall_data["Spending Score (1-100)"],
    c=mall_data["Cluster"],
    cmap="rainbow",
    s=45,
    alpha=0.75
)
plt.scatter(
    mall_centers[:, 0],
    mall_centers[:, 1],
    marker="X",
    s=250,
    c="black",
    edgecolor="white",
    label="Centroids"
)
plt.title("Mall Customers - Customer Segmentation")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# ============================================================
# PART 4: Bonus - XGBoost GridSearchCV
# ============================================================

print("\n" + "=" * 60)
print("BONUS: XGBOOST HYPERPARAMETER TUNING")
print("=" * 60)

try:
    from xgboost import XGBRegressor

    diabetes = load_diabetes()
    X_bonus = diabetes.data
    y_bonus = diabetes.target

    X_train, X_test, y_train, y_test = train_test_split(
        X_bonus,
        y_bonus,
        test_size=0.2,
        random_state=7
    )

    xgb_model = XGBRegressor(
        objective="reg:squarederror",
        random_state=7,
        n_jobs=1
    )

    param_grid = {
        "max_depth": [2, 3, 4],
        "n_estimators": [50, 100, 150]
    }

    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        cv=3,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    print("\nBest XGBoost parameters:")
    print(grid_search.best_params_)
    print(f"Best cross-validation RMSE: "
          f"{np.sqrt(-grid_search.best_score_):.2f}")

    best_xgb_model = grid_search.best_estimator_
    test_score = best_xgb_model.score(X_test, y_test)
    print(f"Test R-squared: {test_score:.3f}")

except ImportError:
    print("XGBoost is not installed. Install it using:")
    print("pip install xgboost")
except Exception as error:
    print(f"XGBoost bonus could not run: {error}")


# ============================================================
# CONCEPT SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("CONCEPT SUMMARY")
print("=" * 60)
print("K-Means: Groups data points into k clusters using centroids.")
print("Elbow method: Uses inertia to help identify a suitable k.")
print("Silhouette score: Measures how well-separated the clusters are.")
print("Customer segmentation: Groups customers by income and spending.")
print("GridSearchCV: Tests parameter combinations using cross-validation.")
