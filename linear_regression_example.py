import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

inputdata = pd.read_csv("car details v4.csv")

print(inputdata)
#The following line is necessary to drop the NaN lines from the dataset:
inputdata = inputdata.dropna()
#y is the predicted value, corresponding to car price
y = inputdata.iloc[:, 2].values
#X is the predictor, corresponding to car's Fuel Tank Capacity (consequently,
#this is related to car's size)
X = inputdata.iloc[:,19].values
print(X)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 3/4)
#The following is necessary for SKlearn function LinearRegression()
#Indeed, X_train and X_test need to be 2D np arrays (with a single feature)
X_train = X_train.reshape(-1,1)
X_test = X_test.reshape(-1,1)

ml_model = LinearRegression()
ml_model.fit(X_train, y_train)

y_pred = ml_model.predict(X_test)


plt.scatter(X_test,y_test,c='orange', label = 'Car Price (y_test)')
plt.plot(X_test,ml_model.predict(X_test), c = "violet", label = 'Linear regression line')
plt.grid(True)
plt.xlabel('Fuel Tank Capacity (X_test)')
plt.ylabel('Car Price')
plt.title('Linear Regression prediction')
plt.legend(loc="upper right")
plt.show()
