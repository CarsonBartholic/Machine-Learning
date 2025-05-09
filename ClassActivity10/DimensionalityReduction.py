import pandas as pd
import numpy as np
import umap
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# load the cancer data from sklearn using load_cancer_data
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
target = pd.Series(data.target)
data = pd.DataFrame(data.data, columns=data.feature_names)
data.index = data.index.astype(str)


y = target
X = data

# Initialize UMAP
xumap = umap.UMAP(min_dist=0.5, n_components=2, random_state=42).fit_transform(X, y)

# Initialize PCA
pca = PCA(n_components=2).fit_transform(X)

# Initialize tSNE
tSNE = TSNE(n_components=2, random_state=42).fit_transform(X)

# Create a plot with three subplots, one for each dimensionality reduction method
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# UMAP plot
axes[0].scatter(x=xumap[:, 0], y=xumap[:, 1], c=y, cmap='Spectral', s=5)
axes[0].set_title('UMAP')

# PCA plot
axes[1].scatter(x=pca[:, 0], y=pca[:, 1], c=y, cmap='Spectral', s=5)
axes[1].set_title('PCA')

# tSNE plot
axes[2].scatter(x=tSNE[:, 0], y=tSNE[:, 1], c=y, cmap='Spectral', s=5)
axes[2].set_title('tSNE')

# Adjust layout and save the combined plot
plt.tight_layout()
plt.savefig('dimensionalityReductionComparison.png')  # Save the combined plot as a PNG file
plt.show()