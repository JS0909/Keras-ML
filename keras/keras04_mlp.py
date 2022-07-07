# multilayer percentron

# 어떤 데이터? = 열, 피처, 특성, 칼럼
# 해당 데이터의 갯수 = 행

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10],
              [1,1,1,1,2,1.3,1.4,1.5,1.6,1.4]])
y = np.array([11,12,13,14,15,16,17,18,19,20])

print(x.shape) # (2, 10)

# x = x.transpose()    =    x = np.transpose(x) // x의 행열 변환
x = x.T # x의 행열 변환
print(x.shape) # (10, 2)
print(x)
print(y.shape) # (10,)


# 2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=2)) # 행 무시, 열 우선, x의 열(특성, 피처, 칼럼)
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(4))
model.add(Dense(1)) # y의 열 = output layer

# 3. 컴파일 훈련
model.compile(loss = 'mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1)

# 4. 평가, 예측
loss = model.evaluate(x,y)
print('loss:', loss)

# result = model.predict([10, 1.4]) # (2, ) // 행열 안바꿨을때
# expected min_ndim=2, found ndim=1. Full shape received: (None,)
result = model.predict([[10, 1.4]]) # (N , 2) <= 데이터 쉐이프 맞춰야함
print('10과 1.4의 예측값은 : ', result)


# model.add(Dense(5, input_dim=2))
# model.add(Dense(4))
# model.add(Dense(3))
# model.add(Dense(4))
# model.add(Dense(1))
# epochs=100, batch_size=1
# loss: 1.3030
# 10, 1.4 예측값 = 20.036375