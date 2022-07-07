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

x_train, x_test, y_train, y_test =  train_test_split(x, y, train_size=0.7, shuffle=True, random_state=66)
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
model.add(Dense(20, activation='relu', input_dim=8))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(70, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam', metrics=['mae'])

from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=30, mode='min', verbose=1, restore_best_weights=True)
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=500, batch_size=50,
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
# loss :  [0.4794151186943054, 0.5155203938484192]
# r2스코어 :  0.6506154419253785
# 걸린 시간:  41.14746689796448

# 2. MinMax scaler
# loss :  [0.2764192521572113, 0.3480816185474396]
# r2스코어 :  0.7985531586672475
# 걸린 시간:  303.28535294532776

# 3. Standard scaler
# loss :  [0.28278881311416626, 0.353474497795105]
# r2스코어 :  0.793911238986394
# 걸린 시간:  125.68351101875305

# 4. MaxAbsScaler
# loss :  [0.3552728593349457, 0.4186839163303375]
# r2스코어 :  0.7410868249055147
# 걸린 시간:  53.814637660980225

# 5. RobustScaler
# loss :  [0.3001979887485504, 0.37224072217941284]
# r2스코어 :  0.7812239148827186
# 걸린 시간:  19.09451651573181