import numpy as np
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error

# y_true = np.array([100.,2])
# y_pred = np.array([200.,102])

y_true = np.array([100.,200])
y_pred = np.array([200.,300])       # << 아래꺼가 더 좋다

mae = mean_absolute_error(y_true, y_pred)
print('mae:', mae)
# mae: 100.0
# mae: 100.0

mape = mean_absolute_percentage_error(y_true, y_pred)
print('mape:', mape)
# mape: 25.5
# mape: 0.75 << 이게 더 좋음