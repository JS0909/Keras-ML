from sklearn.metrics import mean_squared_error
import numpy as np

def RMSE(y_test, y_predict): # rmse 계산 사용 법
    return np.sqrt(mean_squared_error(y_test, y_predict))

# rmse = RMSE(y_predict, y_test)
# print('rmse: ', rmse)