from sklearn.datasets import fetch_california_housing
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import datetime
import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

# 1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test =  train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

# 2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=8))
model.add(Dense(50, activation='relu'))
model.add(Dense(30))
model.add(Dense(50, activation='relu'))
model.add(Dense(20))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))


# 3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam', metrics=['mae'])

date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M')
filepath = './_ModelCheckPoint/k25/03/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
earlyStopping = EarlyStopping(monitor='val_loss', patience=30, mode='min', verbose=1, restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1,
                      save_best_only=True, filepath= "".join([filepath, 'k25_',date, '_', filename]))
earlyStopping = EarlyStopping(monitor='val_loss', patience=30, mode='min', verbose=1, restore_best_weights=True)

hist = model.fit(x_train, y_train, epochs=200, batch_size=50,
                callbacks=[earlyStopping, mcp],
                validation_split=0.25)

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)

