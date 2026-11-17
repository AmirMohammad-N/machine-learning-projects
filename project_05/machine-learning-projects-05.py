import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics

df = pd.read_csv("C:/Users/Pc/Desktop/New folder (2)/drug200.csv")
print(df.head())


x = np.asanyarray(df[['Age','Sex','BP','Cholesterol','Na_to_K']])
y = np.asanyarray(df[['Drug']])

t_sex = preprocessing.LabelEncoder()
t_sex.fit(['F','M'])
x[:,1] = t_sex.transform(x[:,1])
print("SEX",x[:,1])

t_bp = preprocessing.LabelEncoder()
t_bp.fit(['LOW','NORMAL','HIGH'])
x[:,2] = t_bp.transform(x[:,2])
print("BP",x[:,2])


t_Chol = preprocessing.LabelEncoder()
t_Chol.fit(['LOW','NORMAL','HIGH'])
x[:,3] = t_Chol.transform(x[:,3])
print("Chol",x[:,3])

x_train , x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=3)

drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
drugTree.fit(x_train , y_train)

predTree = drugTree.predict(x_test)

print (predTree [0:5])
print (y_test [0:5])


print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_test, predTree))



























