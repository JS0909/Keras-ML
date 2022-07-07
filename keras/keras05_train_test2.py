import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

# [실습, 과제] 넘파이 리스트의 슬라이싱!! 7:3으로 잘라라
x_train = x[:7] # index 7직전까지(제외) 7 빼기 하나까지
x_test = x[7:] # index 7부터(포함)
y_train = y[:7]
y_test = y[7:] # 혹은 [7:10]

# print(x[1:-1]) :  index1 ~ index길이-1(=마지막index번호-1)

# print(x_test)
# print(x_train)
# print(y_test)
# print(y_train)

# 2. 모델
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)
result = model.predict([11])
print('[11]의 예측값 : ', result)
