import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN


X, y = make_moons(n_samples=1000,noise=0.08,random_state=42)
plt.scatter(X[:, 0], X[:, 1])
plt.show()

scaler = preprocessing.StandardScaler()
X_scaled = scaler.fit_transform(X)

model = DBSCAN(eps=0.3,min_samples=5)
fit = model.fit_predict(X_scaled)


plt.scatter(X_scaled[:, 0], X_scaled[:, 1],c=fit,cmap='viridis')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('DBSCAN Clustering')

plt.show()

n_clusters = len(set(fit)) - (1 if -1 in fit else 0)

n_noise = list(fit).count(-1)

print("Number of clusters:", n_clusters)
print("Number of noise points:", n_noise)