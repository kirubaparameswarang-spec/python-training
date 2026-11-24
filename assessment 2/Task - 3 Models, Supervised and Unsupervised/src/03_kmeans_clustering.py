import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# 1. Load dataset
df = pd.read_csv("data/loan_preprocessing.csv")


# 2. Select three numeric columns
features = ["annual_income", "loan_amount", "credit_score"]

X = df[features].copy()


# 3. Handle missing values
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)


# 4. Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)


# 5. Run K-Means for k = 2 to 8
inertias = []
silhouette_scores = []

k_values = range(2, 9)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)

    labels = kmeans.fit_predict(X_scaled)

    inertias.append(kmeans.inertia_)

    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)

    print(f"k={k}, " f"Inertia={kmeans.inertia_:.2f}, " f"Silhouette Score={score:.4f}")


# 6. Create elbow and silhouette plots
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(k_values, inertias, marker="o")
plt.title("Elbow Method")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.xticks(list(k_values))
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(k_values, silhouette_scores, marker="o")
plt.title("Silhouette Scores")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.xticks(list(k_values))
plt.grid(True)

plt.tight_layout()

plt.savefig("outputs/elbow_silhouette.png", dpi=300)

plt.show()


# 7. Select k with the highest silhouette score
best_k = list(k_values)[silhouette_scores.index(max(silhouette_scores))]

print("\nBEST NUMBER OF CLUSTERS:", best_k)


# 8. Fit final K-Means model
final_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)

df["cluster"] = final_kmeans.fit_predict(X_scaled)


# 9. Describe customer segments
print("\nCUSTOMER SEGMENT SIZES")
print(df["cluster"].value_counts().sort_index())

print("\nCUSTOMER SEGMENT PROFILES")

segment_summary = df.groupby("cluster")[features].mean().round(2)

print(segment_summary)

segment_summary.to_csv("outputs/cluster_segment_summary.csv")


# 10. Save dataset with cluster labels
df.to_csv("outputs/loan_customer_clusters.csv", index=False)

print("\nCLUSTERING TASK COMPLETED")
print("Saved: outputs/elbow_silhouette.png")
print("Saved: outputs/cluster_segment_summary.csv")
print("Saved: outputs/loan_customer_clusters.csv")
