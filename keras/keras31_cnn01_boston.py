from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
import numpy as np
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, MaxPool2D, Conv2D, Flatten
from sklearn.metrics import r2_score
import pandas as pd

# 1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets['target']
print(x.shape)

x_train, x_test, y_train, y_test =  train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)
print(x_train.shape, y_train.shape) # (404, 13) (404,)
print(x_test.shape, y_test.shape) # (102, 13) (102,)

# ------스케일링------
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
# --------------------

x_train = x_train.reshape(404, 13, 1, 1)
x_test = x_test.reshape(102, 13, 1, 1)
print(x_train.shape)

# 2. 모델구성
model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(1,1), strides=1, padding='same', input_shape=(13, 1, 1)))
model.add(MaxPool2D())
model.add(Conv2D(10, (1,1),padding='valid', activation='relu'))
model.add(Conv2D(5, (1,1),padding='same', activation='relu'))
model.add(Conv2D(4, (1,1),padding='valid', activation='relu'))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='softmax'))
model.summary()

# 3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam', metrics=['mae'])

from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=30, mode='min', verbose=1, restore_best_weights=True)

hist = model.fit(x_train, y_train, epochs=200, batch_size=64,
                 validation_split=0.2,
                 callbacks=[earlyStopping],
                 verbose=1)

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss: ', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2:', r2)

# DNN
# loss:  [8.3637056350708, 2.1401515007019043]
# r2: 0.8999352707266075
