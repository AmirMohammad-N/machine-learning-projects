import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv("C:/Users/Pc/Desktop/New folder (2)/FuelConsumption.csv")

cdf = df[["ENGINESIZE","CYLINDERS","FUELCONSUMPTION_COMB","CO2EMISSIONS"]]

np.random.seed(42)
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

regr = linear_model.LinearRegression()

train_x = np.asanyarray(train[["ENGINESIZE","CYLINDERS","FUELCONSUMPTION_COMB"]])
train_y = np.asanyarray(train[["CO2EMISSIONS"]])


test_x = np.asanyarray(test[["ENGINESIZE","CYLINDERS","FUELCONSUMPTION_COMB"]])
test_y = np.asanyarray(test[["CO2EMISSIONS"]])

regr.fit(train_x, train_y)
print(regr.coef_)
print(regr.intercept_)




predictions_train = regr.predict(train_x)
predictions_test = regr.predict(test_x)


plt.scatter(train_y, predictions_train, color='blue')
plt.suptitle("Train")
plt.xlabel("Actual Emission")
plt.ylabel("Predicted Emission")
plt.show()


plt.scatter(test_y, predictions_test, color='blue')
plt.suptitle("Test")
plt.xlabel("Actual Emission")
plt.ylabel("Predicted Emission")
plt.show()


mae = mean_absolute_error(test_y, predictions_test)
r2 = r2_score(test_y, predictions_test)

print("MAE:", mae)
print("R2 Score:", r2)