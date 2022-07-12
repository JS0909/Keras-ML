from sklearn.datasets import fetch_california_housing
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np

# 1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test =  train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)
# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler() 
# scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(np.min(x_train)) # 0.0
print(np.max(x_train)) # 1.0
print(np.min(x_test))
print(np.max(x_test))

# 2. 모델구성
# 시퀀셜
# model = Sequential()
# model.add(Dense(5, input_dim=8))
# model.add(Dense(50, activation='relu'))
# model.add(Dense(30))
# model.add(Dense(50, activation='relu'))
# model.add(Dense(20))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(1))

# 함수
input1 = Input(shape=(8,))
dense1 = Dense(5)(input1)
dense2 = Dense(50, activation='relu')(dense1)
dense3 = Dense(30)(dense2)
dense4 = Dense(50, activation='relu')(dense3)
dense5 = Dense(20)(dense4)
dense6 = Dense(10, activation='relu')(dense5)
output1 = Dense(1)(dense6)
model = Model(inputs=input1,outputs=output1)

# 3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam', metrics=['mae'])

from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=30, mode='min', verbose=1, restore_best_weights=True)
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=200, batch_size=50,
                callbacks=[earlyStopping],
                validation_split=0.25)
end_time = time.time()

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)
print('걸린 시간: ', end_time-start_time)

# StandardScaler
# loss :  [0.2475900799036026, 0.32575559616088867]
# r2스코어 :  0.8223970679831727
# 걸린 시간:  37.20634174346924