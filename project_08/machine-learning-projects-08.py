import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=2000,centers=3,n_features=2,cluster_std=1.5,random_state=42)

scaler = preprocessing.StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_scaled)

plt.scatter(X_scaled[:, 0],X_scaled[:, 1],c=kmeans.labels_,cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0],kmeans.cluster_centers_[:, 1],marker='X',s=200,color='red')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('K-Means Clustering')
plt.show()


inertias = []

for k in range(1, 10):
    kmeans = KMeans(n_clusters=k,random_state=42,n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)


plt.plot(range(1, 10), inertias, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method')

plt.show()
