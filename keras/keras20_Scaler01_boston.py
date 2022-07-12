from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.metrics import r2_score
import time

# 1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets['target']

# print(np.min(x)) # 0.0
# print(np.max(x)) # 711.0
# x = (x - np.min(x)) / (np.max(x) - np.min(x))
# print(x[:10])
# 잘못됨

# ------스케일링------
scaler = MinMaxScaler() # 각 열에서 가장 작은 값을 0, 큰 값을 1로 잡고 나머지를 비율로 표시
# scaler = StandardScaler() # 평균을 0으로 잡고 표준편차로 나눠줌
# scaler = MaxAbsScaler() # 각 열에서 가장 큰 값을 1로 잡고 나머지를 비율로 표시
# scaler = RobustScaler() #  25th~75th의 데이터와 중앙값으로 표준 정규화 (standard scaling)

x_train, x_test, y_train, y_test =  train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
x_train = scaler.fit_transform(x_train) # 위에 두줄이 한줄로 가능
x_test = scaler.transform(x_test)
print(np.min(x_train)) # 0.0
print(np.max(x_train)) # 1.0
print(np.min(x_test))
print(np.max(x_test))
# --------------------

# 2. 모델구성
model = Sequential()
model.add(Dense(50, activation='relu', input_dim=13))
model.add(Dense(100, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1)) # relu는 마지막 out put에 넣으면 안됨

# 3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam', metrics=['mae'])

from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=30, mode='min', verbose=1, restore_best_weights=True)
# monitor에서 제공하는 것은 val_loss, loss 정도이고 R2는 제공 안함
# mode auto면 loss계열은 자동으로 최소, accuracy계열은 자동으로 최대 찾음

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

# 보스턴에 대한 3가지 비교
# 1. 스케일러 하기 전
# 2. MinMax scaler
# 3. Standard scaler

# 1. 스케일러 하기 전
# loss:  [17.222091674804688, 3.1463584899902344]
# r2: 0.7939520472325667
# 걸린 시간:  5.161373138427734

# 2. MinMax scaler
# loss:  [10.335180282592773, 2.2913012504577637]
# r2: 0.8763482194187748
# 걸린 시간:  4.370350360870361

# 3. Standard scaler
# loss:  [7.966587066650391, 2.0470046997070312]
# r2: 0.9046864649014508
# 걸린 시간:  4.291144847869873

# 4. MaxAbsScaler
# loss:  [12.675909042358398, 2.4529151916503906]
# r2: 0.8483433352370704
# 걸린 시간:  2.4608144760131836

# 5. RobustScaler
# loss:  [8.869993209838867, 2.3156251907348633]
# r2: 0.8938779271920622
# 걸린 시간:  4.405545473098755