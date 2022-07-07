import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. 데이터
x = np.array([range(10), range(21, 31), range(201,211)])

print(x)
print(x.shape)

y = np.array([[1,2,3,4,5,6,7,8,9,10],
              [1.,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9],
              [9,8,7,6,5,4,3,2,1,0]])
print(y.shape)

x = x.T
y = y.T
print(x.shape)
print(y.shape)

# 2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=3)) 
model.add(Dense(4))
model.add(Dense(5))
model.add(Dense(30))
model.add(Dense(10))
model.add(Dense(50))
model.add(Dense(5))
model.add(Dense(3))

# 3. 컴파일 훈련
model.compile(loss = 'mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1)

# 4. 평가, 예측
loss = model.evaluate(x,y)
print('loss:', loss)
result = model.predict([[9, 30, 210]])
print('9, 30, 210의 예측값은 : ', result) # 10, 1.9, 0

# model.add(Dense(5, input_dim=3)) 
# model.add(Dense(4))
# model.add(Dense(5))
# model.add(Dense(30))
# model.add(Dense(10))
# model.add(Dense(50))
# model.add(Dense(5))
# model.add(Dense(3))
# epochs=100, batch_size=1
# lose: 5.175589194550412e-06
# 9, 30, 210의 예측값은 9.9990053e+00 .0.9029000e+00 6.1596266e-04

# 11.6e+05 1100000.6
# 11e-05 0.000011