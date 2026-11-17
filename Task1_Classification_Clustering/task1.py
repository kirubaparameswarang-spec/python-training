from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.cluster import KMeans
import numpy as np

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="macro")
cm = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f} | F1 (macro): {f1:.2f}")
print("Confusion Matrix:")
print(cm)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X)

cluster_sizes = np.bincount(clusters)

#print("\n--- KMEANS CLUSTERING RESULTS ---")
print("KMeans cluster sizes:", cluster_sizes.tolist())

# 5. Compare clusters with actual species
#print("\n--- CLUSTER VS ACTUAL SPECIES ---")

#for i in range(3):
    #print(f"\nCluster {i}:")
    #for j in range(3):
        #count = np.sum((clusters == i) & (y == j))
        #print(f"{iris.target_names[j]}: {count}")

# 6. Observations
#print("\n--- OBSERVATIONS ---")
#print("1. KMeans groups similar flowers based on their features.")
#print("2. Setosa is generally separated clearly from other species.")
#print("3. Versicolor and Virginica have some overlapping features.")

# 7. Supervised vs Unsupervised
#print("\n--- SUPERVISED VS UNSUPERVISED ---")
#print("1. Logistic Regression needs labelled data to learn flower species.")
#print("2. KMeans does not need labelled data because it groups flowers by similarity.")
