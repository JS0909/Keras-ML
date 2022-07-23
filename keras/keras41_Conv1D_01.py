import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, SimpleRNN, LSTM, Conv1D, Flatten
from keras.preprocessing import sequence
from tensorflow.python.keras.callbacks import EarlyStopping
import tensorflow as tf
import time

# 1. 데이터
dataset = np.array([1,2,3,4,5,6,7,8,9,10])

x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9]])
y = np.array([4,5,6,7,8,9,10])
print(x.shape, y.shape) # (7, 3) (7,)

x = x.reshape(7, 3, 1)
print(x.shape) # (7, 3, 1)

# 2. 모델구성
model = Sequential()
model.add(Conv1D(10, 2, input_shape=(3,1))) # Conv1D는 3차원 먹고 3차원 뱉음
# model.add(LSTM(10, input_shape=(3,1))) # LSTM은 3차원 먹고 2차원 뱉음
model.add(Flatten())
model.add(Dense(10, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(1))

model.summary() # LSTM params: 480 // Conv1D params: 30

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
start_time = time.time()
Es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=500, restore_best_weights=True)
model.fit(x,y, epochs=5000, callbacks=[Es], validation_split=0.1)
end_time = time.time()

# 4. 평가, 예측
loss = model.evaluate(x, y)
y_pred = np.array([8, 9, 10]).reshape(1, 3, 1) # 3차원으로 변경 [[[8], [9],[10]]]
result = model.predict(y_pred)
print('loss: ', loss)
print('결과: ', result)
print('시간: ', end_time-start_time)

# loss:  1.090680939341837e-06
# [8, 9, 10]의 결과:  [[10.950838]]

# Conv1D
# loss:  1.280575270357076e-05
# 결과:  [[11.018893]]
# 시간:  21.064549684524536