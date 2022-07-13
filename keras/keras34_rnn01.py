import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, SimpleRNN, Dropout
from keras.preprocessing import sequence
from tensorflow.python.keras.callbacks import EarlyStopping

# 1. 데이터
dataset = np.array([1,2,3,4,5,6,7,8,9,10])

x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9]])
y = np.array([4,5,6,7,8,9,10])
print(x.shape, y.shape) # (7, 3) (7,)

# RNN의 x_shape = (행, 열, 그냥 몇장인지=CNN의 채널이랑 비슷)
x = x.reshape(7, 3, 1)
print(x.shape) # (7, 3, 1)

# 2. 모델구성
model = Sequential()
model.add(SimpleRNN(64, input_shape=(3,1))) # 행무시 열우선, SimpleRNN은 2차원을 리턴한다
# model.add(SimpleRNN(32)) # ndim 오류, 3차원 받아야되는데 2차원 받아서
model.add(Dense(32, activation='swish'))
model.add(Dense(64, activation='swish'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
Es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100, restore_best_weights=True)
model.fit(x,y, epochs=5000, callbacks=[Es], validation_split=0.1)

# 4. 평가, 예측
loss = model.evaluate(x, y)
y_pred = np.array([8, 9, 10]).reshape(1, 3, 1) # 3차원으로 변경 [[[8], [9],[10]]]
result = model.predict(y_pred)
print('loss: ', loss)
print('[8, 9, 10]의 결과: ', result)

# loss:  1.090680939341837e-06
# [8, 9, 10]의 결과:  [[10.950838]]