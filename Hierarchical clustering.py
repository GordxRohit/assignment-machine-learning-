import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

X = np.array([[10], [12], [14], [40], [42], [44]])
names = ["C1", "C2", "C3", "C4", "C5", "C6"]

Z = linkage(X, method="ward")

labels = fcluster(Z, t=2, criterion="maxclust")

print("Customer cluster assignments:")
for name, value in zip(names, labels):
    print(f"{name}: Cluster {value}")

plt.figure(figsize=(8, 4))
dendrogram(Z, labels=names)
plt.title("Hierarchical Clustering - Ward Linkage")
plt.xlabel("Customers")
plt.ylabel("Ward Distance")
plt.tight_layout()
plt.show()