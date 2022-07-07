from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

# 1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,3,4,5,7,9,3,8,12,13,8,14,15,9,6,17,23,21,20])

# train_test_split
x_train, x_test, y_train, y_test =  train_test_split(x, y, train_size=0.7, shuffle=True, random_state=66)


# 2. 모델구성
model = Sequential()
model.add(Dense(7, input_dim=1))
model.add(Dense(10))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(7))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam')
model.fit(x_train, y_train, epochs=400, batch_size=1)


# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x)
# x 전체에 대한 y 예상값, x_test를 넣을 경우 슬라이싱 했던 데이터만 가지고 y의 예상 값을 반환함

import matplotlib.pyplot as plt # 그래프 그리는 API, 시각화

plt.scatter(x, y) # 점뿌리기
plt.plot(x, y_predict, color='red') # 선그리기
plt.show() # 보여주기