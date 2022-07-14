import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Bidirectional, LSTM
from keras.preprocessing import sequence
from tensorflow.python.keras.callbacks import EarlyStopping
import tensorflow as tf

# 1. 데이터
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
              [5,6,7],[6,7,8],[7,8,9],[8,9,10],
              [9,10,11],[10,11,12],
              [20,30,40],[30,40,50],[40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])

x_predict = np.array([50, 60, 70])
x = x.reshape(13, 3, 1)

# 2. 모델구성
model = Sequential()
model.add(Bidirectional(LSTM(100, return_sequences=True), input_shape=(3,1)))
model.add(LSTM(100, activation='relu'))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
Es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100, restore_best_weights=True)
model.fit(x,y, epochs=1000, callbacks=[Es], validation_split=0.1)

# 4. 평가, 예측
loss = model.evaluate(x, y)
y_pred = x_predict.reshape(1, 3, 1)
result = model.predict(y_pred) # 프레딕트는 위의 모델에 쓰는 거나 다름없으니까 훈련데이터랑 인풋 똑같아야됨
print('loss: ', loss)
print('[50, 60, 70]의 결과: ', result)

