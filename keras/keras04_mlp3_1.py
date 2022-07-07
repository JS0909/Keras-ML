import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. 데이터
x = np.array([range(10), range(21, 31), range(201,211)]) # 0부터 10 전까지, 21부터 31 전 까지
# range(a) // 0 ~ a-1 까지 정수를 array 형태로 반환
# range(a, b) // a ~ b-1 까지 정수를 array 형태로 반환

print(x)
print(x.shape) # (3, 10)

y = np.array([[1,2,3,4,5,6,7,8,9,10],
              [1.,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9]])
print(y.shape) # (2, 10)

x = x.T
y = y.T
print(x.shape) # (10, 3)
print(y.shape) # (10, 2)

# 2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=3)) 
model.add(Dense(4))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(4))
model.add(Dense(5))
model.add(Dense(2))

# 3. 컴파일 훈련
model.compile(loss = 'mse', optimizer='adam')
model.fit(x, y, epochs=200, batch_size=1)

# 4. 평가, 예측
loss = model.evaluate(x,y)
print('loss:', loss)
result = model.predict([[9, 30, 210]])
print('9, 30, 210의 예측값은 : ', result) # [[10, 1.9]]에 가까워야함

# 예측: [[9, 30, 210]] -> 예상 y값 [[10, 1.9]]

# model.add(Dense(5, input_dim=3))
# model.add(Dense(4))
# model.add(Dense(4))
# model.add(Dense(3))
# model.add(Dense(4))
# model.add(Dense(5))
# model.add(Dense(2))
# epochs=200, batch_size=1
# loss: 8.756815077504143e-07
# 9, 30, 210의 예측값은 :  [[9.999871  1.9018754]]