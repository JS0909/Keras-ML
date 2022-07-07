from sklearn.datasets import fetch_california_housing
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
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
# scaler = StandardScaler()
# scaler = MaxAbsScaler() 
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(np.min(x_train)) # 0.0
print(np.max(x_train)) # 1.0
print(np.min(x_test))
print(np.max(x_test))

# 2. 모델구성
model = Sequential()
model.add(Dense(5, activation='relu', input_dim=8))
model.add(Dense(50, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

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

# 1. 스케일러 하기 전
# loss :  [0.48650550842285156, 0.508192777633667]
# r2스코어 :  0.6510165780474442
# 걸린 시간:  218.77557635307312

# 2. MinMax scaler
# loss :  [0.28370004892349243, 0.3621354401111603]
# r2스코어 :  0.7964943852676224
# 걸린 시간:  241.7563350200653

# 3. Standard scaler
# r2스코어 :  0.7966497663161433
# 걸린 시간:  142.34565138816833

# 4. MaxAbsScaler
# loss :  [0.38276439905166626, 0.42861926555633545]
# r2스코어 :  0.7254329473819731
# 걸린 시간:  43.33633804321289

# 5. RobustScaler
# loss :  [0.3149472177028656, 0.37359118461608887]
# r2스코어 :  0.7740799206616391
# 걸린 시간:  35.48738121986389