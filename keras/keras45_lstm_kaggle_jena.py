# RNN, CNN 1개 이상씩 사용
import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, GRU, Conv1D, Flatten, Reshape
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
import tensorflow as tf
import time
import datetime as dt
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# 1. 데이터
path = './_data/kaggle_jena/'
datasets = pd.read_csv(path + 'jena_climate_2009_2016.csv')

# print(datasets.info())
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 420551 entries, 0 to 420550
# Data columns (total 15 columns):
 #   Column           Non-Null Count   Dtype
# ---  ------           --------------   -----
#  0   Date Time        420551 non-null  object
#  1   p (mbar)         420551 non-null  float64
#  2   T (degC)         420551 non-null  float64
#  3   Tpot (K)         420551 non-null  float64
#  4   Tdew (degC)      420551 non-null  float64
#  5   rh (%)           420551 non-null  float64
#  6   VPmax (mbar)     420551 non-null  float64
#  7   VPact (mbar)     420551 non-null  float64
#  8   VPdef (mbar)     420551 non-null  float64
#  9   sh (g/kg)        420551 non-null  float64
#  10  H2OC (mmol/mol)  420551 non-null  float64
#  11  rho (g/m**3)     420551 non-null  float64
#  12  wv (m/s)         420551 non-null  float64
#  13  max. wv (m/s)    420551 non-null  float64
#  14  wd (deg)         420551 non-null  float64
# dtypes: float64(14), object(1)

datasets['Date Time'] = pd.to_datetime(datasets['Date Time'])
datasets['year'] = datasets['Date Time'].dt.strftime('%Y') 
datasets['month'] = datasets['Date Time'].dt.strftime('%m')
datasets['hour'] = datasets['Date Time'].dt.strftime('%H')
datasets['date'] = datasets['Date Time'].dt.strftime('%d')
datasets = datasets.drop(['Date Time'], axis=1)

cols = ['month', 'date', 'year', 'hour']
for col in cols:
    le = LabelEncoder()
    datasets[col] = le.fit_transform(datasets[col])

# datasets.shape (n, 18)

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        aaa.append(subset) # 하나하나 리스트 형태로 추가함
    return np.array(aaa)

bbb = split_x(datasets, 5)
# print(bbb)
print(bbb.shape) # (420547, 5, 18)

x =  bbb[:, :-1]
y =  bbb[:, -1]
# print(x, y)
print(x.shape, y.shape) # (420547, 4, 15) (420547, 15)

x_train, x_test, y_train, y_test =  train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

# 2. 모델구성
model = Sequential()
model.add(Conv1D(1, 2, input_shape=(4,18)))
model.add(GRU(1))
model.add(Dense(1, activation = 'relu'))
model.add(Dense(1, activation = 'relu'))
model.add(Dense(1, activation = 'relu'))
model.add(Dense(1))

# model.summary()

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
start_time = time.time()
Es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=500, restore_best_weights=True)
model.fit(x_train, y_train, epochs=1, callbacks=[Es], validation_split=0.1)
end_time = time.time()

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)

