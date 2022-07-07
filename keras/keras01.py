# 1. 데이터
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
import numpy as np  # 행렬 API
x = np.array([1, 2, 3])
y = np.array([1, 2, 3])

# 2. 모델구성

model = Sequential()
model.add(Dense(3, input_dim=1))  # Dense(출력, input dimention)
model.add(Dense(5))  # 위에서 add 되기 때문에 input 따로 필요 x
model.add(Dense(6)) # 곱연산으로 들어감
model.add(Dense(3))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') # 평균 제곱 오차로 떨어진 거리 계산하는 방식인 mse 방식을 택해서 loss를 뽑겠다
model.fit(x, y, epochs=250)  # epochs 만큼 훈련하겠다

# fit : (맞도록) 훈련시켜라 (피트니스 생각 하면 쉬움)

# 4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)

result = model.predict([4])
print('[4]의 예측값 : ', result)

