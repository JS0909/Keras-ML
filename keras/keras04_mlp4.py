import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. 데이터
x = np.array([range(10)]) # (1, 10)

# print(x)
# print(x.shape)

y = np.array([[1,2,3,4,5,6,7,8,9,10],
              [1.,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9],
              [9,8,7,6,5,4,3,2,1,0]])
print(y.shape)

# x = x.T
# y = y.T
# print(x.shape)
# print(y.shape)

# 2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=1)) 
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
loss = model.evaluate(x,y) # 과적합 된다, 훈련 데이터와 평가 데이터가 같기 때문
                            # 그래서 70프로는 훈련데이터로, 30프로는 평가데이터로 사용
print('loss:', loss)
result = model.predict([[9]])
print('9의 예측값은 : ', result)

# model.add(Dense(5, input_dim=1)) 
# model.add(Dense(4))
# model.add(Dense(5))
# model.add(Dense(30))
# model.add(Dense(10))
# model.add(Dense(50))
# model.add(Dense(5))
# model.add(Dense(3))
# epochs=100, batch_size=1
# loss = 4.1537883159123434e-13
# 예측값 = 1.0000000e+01, 1.9000002e+00, 5.9604645e-07
# 10.0000000, 1.9000002, 0.00000059604645
