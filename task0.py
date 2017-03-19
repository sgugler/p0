from sklearn.metrics import mean_squared_error
import numpy as np
import getData

(X, y) = getData.loadTrain()
y_test = getData.loadTest()

print(X)

# RMSE = mean_squared_error(y, y_pred) ** 0.5

# print(RMSE)
