import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score

# 1. Load dataset
iris = load_iris()
X = iris.data
y = iris.target

print("Dataset shape:", X.shape)
print("Features:", iris.feature_names)

# 2. Normalize features (K-Means distance par kaam karta hai, isliye scaling zaroori hai)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Apply K-Means
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)
centroids = kmeans.cluster_centers_

print("\nCluster counts:")
for i in range(3):
    print(f"Cluster {i}: {np.sum(clusters == i)}")

# 4. Evaluate with Adjusted Rand Index (ARI)
ari = adjusted_rand_score(y, clusters)
print(f"\nAdjusted Rand Index: {round(ari, 3)}")
# ARI 0.62 aayega, jo batata hai ki clusters actual species se kafi match karte hain.

# 5. Visualize (Petal Length vs Petal Width)
plt.figure(figsize=(8, 5))
plt.scatter(X[:, 2], X[:, 3], c=clusters, cmap="viridis", s=50, edgecolor="black", label="Data points")

# Plot centroids (inverse transform to get original scale)
centroids_original = scaler.inverse_transform(centroids)
plt.scatter(centroids_original[:, 2], centroids_original[:, 3], marker="X", s=250, c="red", edgecolor="black", label="Cluster centroids")

plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.title("K-means Clustering on Iris Dataset (K=3)")
plt.legend()
plt.tight_layout()
plt.show()
