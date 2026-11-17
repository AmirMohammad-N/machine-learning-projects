import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn import preprocessing
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import make_blobs


X, y = make_blobs(n_samples=2000,centers=4,n_features=2,cluster_std=2,random_state=42)

plt.scatter(X[:, 0], X[:, 1])
plt.xlabel('X')
plt.ylabel('Y')

plt.show()

scaler = preprocessing.StandardScaler()
X_scaled = scaler.fit_transform(X)

Z = linkage(X_scaled, method="ward")

plt.figure(figsize=(12, 6))

dendrogram(Z)

plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Data Points")
plt.ylabel("Distance")

plt.show()


model = AgglomerativeClustering(n_clusters=4,linkage="ward")
labels = model.fit_predict(X)

plt.scatter(X[:, 0],X[:, 1],c=labels,cmap="rainbow")

plt.title("Hierarchical Clustering")

plt.xlabel("Annual Income")
plt.ylabel("Spending Score")

plt.show()