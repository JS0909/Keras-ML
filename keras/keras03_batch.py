# naming rule
# kaggle_bike -> python
# kaggleBike -> Java (camel case)

# batch : 분할해서 데이터 넣기

# hyper parameter tuning
# 노드 갯수, 레이어 깊이, 훈련 횟수, 분할

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
import numpy as np

# 1. 데이터
x = np.array([1, 2, 3, 5, 4])
y = np.array([1, 2, 3, 4, 5])

# 2. 모델구성 
model = Sequential()
model.add(Dense(70, input_dim=1))
model.add(Dense(60))
model.add(Dense(70))
model.add(Dense(90))
model.add(Dense(80))
model.add(Dense(50))
model.add(Dense(70))
model.add(Dense(80))
model.add(Dense(70))
model.add(Dense(60))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1, batch_size=1)


# 4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)

result = model.predict([6])
print('[6]의 예측값 : ', result)


#model.add(Dense(80, input_dim=1))
# model.add(Dense(50))
# model.add(Dense(90))
# model.add(Dense(80))
# model.add(Dense(70))
# model.add(Dense(80))
# model.add(Dense(70))
# model.add(Dense(1))
# epochs=1, batch_size=1
# 결과치 : 5.9506383
