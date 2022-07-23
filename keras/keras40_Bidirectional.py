import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Bidirectional
from keras.preprocessing import sequence
from tensorflow.python.keras.callbacks import EarlyStopping
import tensorflow as tf

# 1. 데이터
dataset = np.array([1,2,3,4,5,6,7,8,9,10])

x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9]])
y = np.array([4,5,6,7,8,9,10])
print(x.shape, y.shape) # (7, 3) (7,)

# RNN의 x_shape = (전체 몇장인지, 행, 열)
x = x.reshape(7, 3, 1)
print(x.shape) # (7, 3, 1)

# 2. 모델구성
model = Sequential()
model.add(SimpleRNN(units=100, return_sequences=True, input_shape=(3,1)))
# model.add(Bidirectional(SimpleRNN(5)))
model.add(Dense(3, activation = 'relu'))
model.add(Dense(1))
model.summary() # bidirectional의 param 연산량은 wrapping한 RNN모델 연산량의 두배, 왕복이니까

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
Es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100, restore_best_weights=True)
model.fit(x, y, epochs=5000, callbacks=[Es], validation_split=0.1)

# 4. 평가, 예측
loss = model.evaluate(x, y)
y_pred = np.array([8, 9, 10]).reshape(1, 3, 1) # 3차원으로 변경 [[[8], [9],[10]]]
result = model.predict(y_pred)
print('loss: ', loss)
print('결과: ', result)

# loss:  1.090680939341837e-06
# [8, 9, 10]의 결과:  [[10.950838]]