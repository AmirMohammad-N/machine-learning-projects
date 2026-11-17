import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import jaccard_score
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv("C:/Users/Pc/Desktop/New folder (2)/cell_samples.csv")

X = np.asanyarray(df[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']]).astype(int)
y = np.asanyarray(df['Class']).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)


clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train)

prdict = clf.predict(X_test)


print("Jaccard Score:", jaccard_score(y_test, prdict, pos_label=4))
print("Accuracy:", accuracy_score(y_test, prdict))
print("Confusion Matrix:\n", confusion_matrix(y_test, prdict))