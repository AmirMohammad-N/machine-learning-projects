import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv("C:/Users/Pc/Desktop/New folder (2)/FuelConsumption.csv")

cdf = df[["ENGINESIZE","CO2EMISSIONS"]]

np.random.seed(42)
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]


train_x = np.asanyarray(train[["ENGINESIZE"]])
train_y = np.asanyarray(train[["CO2EMISSIONS"]])


test_x = np.asanyarray(test[["ENGINESIZE"]])
test_y = np.asanyarray(test[["CO2EMISSIONS"]])


poly = PolynomialFeatures(degree=2)
train_x_poly = poly.fit_transform(train_x)


regr = linear_model.LinearRegression()
regr.fit(train_x_poly, train_y)

print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)

plt.scatter(test.ENGINESIZE, test.CO2EMISSIONS,  color='blue')
XX = np.arange(0.0, 10.0, 0.1)
yy = regr.intercept_[0]+ regr.coef_[0][1]*XX+ regr.coef_[0][2]*np.power(XX, 2)
plt.plot(XX, yy, '-r' )
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

test_x_poly = poly.transform(test_x)
predictions = regr.predict(test_x_poly)

print("MAE:", mean_absolute_error(test_y, predictions))
print("R2:", r2_score(test_y, predictions))
