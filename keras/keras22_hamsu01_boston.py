from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
import numpy as np
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input
from sklearn.metrics import r2_score
import time

# 1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets['target']

# ------스케일링------
# scaler = MinMaxScaler() # 각 열에서 가장 작은 값을 0, 큰 값을 1로 잡고 나머지를 비율로 표시
scaler = StandardScaler() # 평균을 0으로 잡고 표준편차로 나눠줌
# scaler = MaxAbsScaler() # 각 열에서 가장 큰 값을 1로 잡고 나머지를 비율로 표시
# scaler = RobustScaler() #  25th~75th의 데이터와 중앙값으로 표준 정규화 (standard scaling)
x_train, x_test, y_train, y_test =  train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(np.min(x_train))
print(np.max(x_train))
print(np.max(x_test))
# --------------------

# 2. 모델구성
# 시퀀셜 모델
# model = Sequential()
# model.add(Dense(50, activation='relu', input_dim=13))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(200, activation='relu'))
# model.add(Dense(200, activation='relu'))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(50, activation='relu'))
# model.add(Dense(1))

# 함수형 모델
input1 = Input(shape=(13,))
dense1 = Dense(50,activation='relu')(input1)
dense2 = Dense(100, activation='relu')(dense1)
dense3 = Dense(200, activation='relu')(dense2)
dense4 = Dense(200, activation='relu')(dense3)
dense5 = Dense(100, activation='relu')(dense4)
dense6 = Dense(50, activation='relu')(dense5)
output1 = Dense(1)(dense6)
model = Model(inputs=input1, outputs=output1)

# 3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam', metrics=['mae'])

from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=30, mode='min', verbose=1, restore_best_weights=True)

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=200, batch_size=64,
                 validation_split=0.2,
                 callbacks=[earlyStopping],
                 verbose=1)
end_time = time.time()

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss: ', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2:', r2)
print('걸린 시간: ', end_time-start_time)

# 시퀀셜 StandardScaler
# loss:  [7.966587066650391, 2.0470046997070312]
# r2: 0.9046864649014508
# 걸린 시간:  4.291144847869873

# StandardScaler
# loss:  [13.795947074890137, 2.576814651489258]
# r2: 0.8349430197820256
# 걸린 시간:  2.9304401874542236

