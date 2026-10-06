import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. Load Iris dataset
iris = load_iris()

X = iris.data
y = iris.target

# 2. Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Apply PCA
pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

# 4. Print explained variance
print("Explained Variance:")
print(pca.explained_variance_ratio_)

# 5. Plot
plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=y
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA on Iris Dataset")

plt.show()