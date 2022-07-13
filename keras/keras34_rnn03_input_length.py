import numpy as np
from tensorboard import summary
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, SimpleRNN, Dropout
from keras.preprocessing import sequence
from tensorflow.python.keras.callbacks import EarlyStopping

# 1. 데이터
dataset = np.array([1,2,3,4,5,6,7,8,9,10])

x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9]])
y = np.array([4,5,6,7,8,9,10])
print(x.shape, y.shape) # (7, 3) (7,)

# RNN의 x_shape = (행, 열, 몇개씩 자르는지=몇번씩연산하는지)
x = x.reshape(7, 3, 1)
print(x.shape) # (7, 3, 1)

# 2. 모델구성
model = Sequential()                                             # input_dim
# model.add(SimpleRNN(10, input_shape=(3,1))) # [batch, timesteps, feature]
model.add(SimpleRNN(units=10, input_length=3, input_dim=1)) # 인풋쉐입을 분리해서 쓸 수 있다
model.add(Dense(5, activation='swish'))
model.add(Dense(1))
model.summary()
# SimpleRNN의 prams = (유닛개수*유닛개수) + (input_dim(feature)수*유닛개수) + (1*유닛개수)
# 그 다음 DNN의 prams = (위의아웃풋유닛개수+1) * 현재노드아웃풋개수

'''
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

'''