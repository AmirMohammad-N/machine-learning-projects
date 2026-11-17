import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import jaccard_score

df = pd.read_csv("C:/Users/Pc/Desktop/New folder (2)/ChurnData.csv")

X = np.asanyarray(df[['tenure', 'age', 'address', 'income','ed', 'employ', 'equip']])

y = np.asanyarray(df['churn']).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

LR = LogisticRegression(C=0.01, solver='liblinear')
LR.fit(X_train, y_train)

yhat = LR.predict(X_test)

print("Jaccard Score:", jaccard_score(y_test, yhat, pos_label=0))
