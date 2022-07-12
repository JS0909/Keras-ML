from sklearn.datasets import load_boston
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from tensorflow.python.keras.callbacks import EarlyStopping

# 1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target
x_train, x_test, y_train, y_test =  train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)
scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler() 
# scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델구성
model = Sequential()
model.add(Dense(64, input_dim=13))
model.add(Dense(32))
model.add(Dense(16, activation='relu'))
model.add(Dense(8))
model.add(Dense(1))

# model.load_weights('./_save/keras23_05_save_weights1.h5') # keras23_05_save_weights 파일에서 랜덤한 가중치 저장시킨 파일
model.load_weights('./_save/keras23_05_save_weights2.h5') # keras23_05_save_weights 파일에서 훈련시켜 구한 가중치 저장시킨 파일
# 컴파일 전에 weight값 불러와서 컴파일 함

# 3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam')
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50, restore_best_weights=True)

# start_time = time.time()
# log = model.fit(x_train, y_train, epochs=100, batch_size=1, callbacks=[es], validation_split=0.2, verbose=1)
# end_time = time.time()

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss: ', loss)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2:', r2)

# weight만 땡기면 모델 구성, 컴파일 다 해줘야됨, 훈련은 다시 안해도 됨
# save_model, load_model, save_weights, load_weights

# 랜덤 가중치 로드
# loss:  598.9628295898438
# r2: -6.166090024397305

# 구한 가중치 로드
# loss:  19.114166259765625
# r2: 0.7713149889990266