import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

X = np.array([[2, 20], [3, 25], [4, 30], [20, 75], [22, 80], [24, 85]])

np.random.seed(42)
random_indices = np.random.choice(len(X), size=2, replace=False)
initial_centroids = X[random_indices]
print("Randomly Initialized Centroids:\n", initial_centroids)

kmeans = KMeans(n_clusters=2, init=initial_centroids, n_init=1, random_state=42)
clusters = kmeans.fit_predict(X)
final_centroids = kmeans.cluster_centers_

print("\nCustomer Cluster Labels:", clusters)

plt.figure(figsize=(6, 5))
plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis', s=100, label='Customers')
plt.scatter(final_centroids[:, 0], final_centroids[:, 1], s=250, c='red', marker='X', label='Final Centroids')
plt.scatter(initial_centroids[:, 0], initial_centroids[:, 1], s=250, c='yellow', marker='*', label='Initial Random Centroids')
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("K-Means Clustering (k=2) with Manual Initialization")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

icsse = 0
for i in range(len(X)):
    cluster_label = clusters[i]
    centroid = final_centroids[cluster_label]
    icsse += np.sum((X[i] - centroid) ** 2)

print(f"\nIntra-Cluster Sum of Squared Error (ICSSE): {round(icsse, 2)}")
