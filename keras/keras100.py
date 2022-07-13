import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import SimpleRNN, Dense
 

dataset = np.array([1, 3, 5, 10, 7, 6, 4])

x = np.array([[1,3,5], [3, 5, 10], [5, 10, 7], [10, 7, 6]])
y = np.array([10, 7, 6, 4])

print(x.shape, y.shape) # (4, 3) (4,)
x = x.reshape(4, 3, 1) # [batch, timesteps, feature]

model = Sequential()
model.add(SimpleRNN(10, input_shape=(3,1))) # 행무시 열우선, SimpleRNN은 2차원을 리턴한다
# model.add(SimpleRNN(32)) # ndim 오류, 3차원 받아야되는데 2차원 받아서
model.add(Dense(32, activation='swish'))
model.add(Dense(64, activation='swish'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))
model.summary()