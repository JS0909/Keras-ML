# RNN, CNN 1개 이상씩 사용
import numpy as np
from sklearn import datasets
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, GRU, Conv1D, Flatten, Reshape
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
import time
import datetime as dt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pandas as pd

# 1. 데이터
path = './_data/kaggle_jena/'
dataset = pd.read_csv(path + 'jena_climate_2009_2016.csv')

# 데이터 정규화
scaler = MinMaxScaler()    
scale_cols = ["p (mbar)","T (degC)","Tpot (K)","Tdew (degC)","rh (%)","VPmax (mbar)",
             "VPact (mbar)","VPdef (mbar)","sh (g/kg)","H2OC (mmol/mol)",
             "rho (g/m**3)","wv (m/s)","max. wv (m/s)","wd (deg)"] # 정규화 대상 칼럼 (Date Time 제외한 나머지 숫자 칼럼)
scaled_dataset = scaler.fit_transform(dataset[scale_cols]) # 정규화 수행
scaled_dataset = pd.DataFrame(scaled_dataset, columns = scale_cols) # 데이터프레임화

# print(dataset.info())
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

dataset['Date Time'] = pd.to_datetime(dataset['Date Time'])
dataset['year'] = dataset['Date Time'].dt.strftime('%Y') 
dataset['month'] = dataset['Date Time'].dt.strftime('%m')
dataset['hour'] = dataset['Date Time'].dt.strftime('%H')
dataset['date'] = dataset['Date Time'].dt.strftime('%d')
dataset = dataset.drop(['Date Time'], axis=1)

cols = ['month', 'date', 'year', 'hour']
for col in cols:
    le = LabelEncoder()
    dataset[col] = le.fit_transform(dataset[col])

target_data = scaled_dataset['T (degC)']
train_data = scaled_dataset[:-200]
target_data = scaled_dataset[-200:]

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        aaa.append(subset)
    return np.array(aaa)

train_data = split_x(train_data, 20)
target_data = split_x(target_data, 20)
# print(bbb)
print(train_data.shape) # (420542, 10, 18)
print(target_data.shape) # (420542, 10)
# bbb = pd.DataFrame(bbb, columns=['month', 'date', 'year', 'hour', "p (mbar)","T (degC)","Tpot (K)","Tdew (degC)","rh (%)",
#                                  "VPmax (mbar)","VPact (mbar)","VPdef (mbar)","sh (g/kg)","H2OC (mmol/mol)",
#                                  "rho (g/m**3)","wv (m/s)","max. wv (m/s)","wd (deg)"])
# 오류 메세지 Must pass 2-d input. shape=(420547, 5, 18)

x_train, x_test, y_train, y_test =  train_test_split(train_data, target_data, train_size=0.8, shuffle=True, random_state=66)

# 2. 모델구성
model = Sequential()
model.add(Conv1D(64, 2, input_shape=(20,18)))
model.add(GRU(128))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(1))

# model.summary()

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
start_time = time.time()
Es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=500, restore_best_weights=True)
fit_log = model.fit(x_train, y_train, epochs=1, batch_size=64, callbacks=[Es], validation_split=0.1)
end_time = time.time()

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
predict = model.predict(x_test)
print('loss: ', loss)
print('prdict: ', predict)

# 그래프
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['font.family'] = 'malgun gothic'
mpl.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(12,6))
plt.plot(y_test, label = 'actual', c = 'red')
plt.plot(predict, c='blue', label='val_loss')
plt.plot(loss, marker = '.', c = 'green', label = 'loss')
plt.grid()
plt.title('예나 온도 예측')
plt.ylabel('온도')
plt.xlabel('날짜')
plt.legend()
plt.show()

# loss:  70.9172134399414
# prdict:  [[9.099443 ]
#  [9.099443 ]
#  [9.099443 ]
#  ...
#  [9.0994425]
#  [9.0994425]
#  [9.0994425]]