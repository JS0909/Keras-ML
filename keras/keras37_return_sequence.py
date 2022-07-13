import numpy as np
from tensorboard import summary
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, LSTM, GRU
from keras.preprocessing import sequence
from tensorflow.python.keras.callbacks import EarlyStopping

# 1. 데이터
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
              [5,6,7],[6,7,8],[7,8,9],[8,9,10],
              [9,10,11],[10,11,12],
              [20,30,40],[30,40,50],[40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x = x.reshape(13, 3, 1)

# 2. 모델구성
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM, Dense

model = Sequential()
model.add(LSTM(10, return_sequences=True, input_shape=(3,1))) # 리턴시퀀스 쓰면 (N, 3, 1) -> (N, 3, 10), 안쓰면 (N, 10)
model.add(LSTM(200, return_sequences=False))
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1)) # Dense는 3차원도 받을 수 있다!
model.summary()

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
Es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5000, restore_best_weights=True)
model.fit(x,y, epochs=500, callbacks=[Es], validation_split=0.1)

# 4. 평가, 예측
loss = model.evaluate(x, y)
x_predict = np.array([50,60,70]).reshape(1,3,1)
result = model.predict(x_predict)
print('loss: ', loss)
print('[8, 9, 10]의 결과: ', result)

# loss:  9.492758750915527
# [8, 9, 10]의 결과:  [[58.969063]]