import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics


df = pd.read_csv("C:/Users/Pc/Desktop/New folder (2)/teleCust1000t.csv")

X = np.asanyarray(df[['region', 'tenure', 'age', 'marital', 'address', 'income', 'ed', 'employ', 'retire', 'gender', 'reside']])
y = np.asanyarray(df['custcat'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

Ks = 25
mean_acc = np.zeros((Ks - 1))

for n in range(1, Ks):
    neigh = KNeighborsClassifier(n_neighbors=n).fit(X_train, y_train)
    yhat = neigh.predict(X_test)
    mean_acc[n - 1] = metrics.accuracy_score(y_test, yhat)

best_k = np.argmax(mean_acc) + 1

print("Best K:", best_k)
print("Best Accuracy:", mean_acc[best_k - 1])

plt.plot(range(1, Ks), mean_acc, marker='o')
plt.xlabel("K")
plt.ylabel("Accuracy")
plt.title("KNN Accuracy K")
plt.xticks(range(1, Ks))
plt.grid()
plt.show()

neigh = KNeighborsClassifier(n_neighbors=best_k)
neigh.fit(X_train, y_train)

yhat = neigh.predict(X_test)

print("Train Accuracy:", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test Accuracy:", metrics.accuracy_score(y_test, yhat))
