import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.decomposition import PCA as SklearnPCA

from models.pca import PCA


X, y = load_iris(return_X_y=True)

my_pca = PCA(n_components=2)
X_my = my_pca.fit_transform(X)

sk_pca = SklearnPCA(n_components=2)
X_sk = sk_pca.fit_transform(X)

print("My explained variance ratio:")
print(my_pca.explained_variance_ratio_)

print("Sklearn explained variance ratio:")
print(sk_pca.explained_variance_ratio_)

print(
    "My total explained variance:",
    np.sum(my_pca.explained_variance_ratio_)
)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_my[:, 0], X_my[:, 1], c=y)
plt.title("PCA from Scratch")

plt.subplot(1, 2, 2)
plt.scatter(X_sk[:, 0], X_sk[:, 1], c=y)
plt.title("sklearn PCA")

plt.tight_layout()
plt.show()