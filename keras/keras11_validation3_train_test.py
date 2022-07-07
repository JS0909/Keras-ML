from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split


# 1. 데이터
x = np.array(range(1,17))
y = np.array(range(1,17))
# [실습] train_test_split로만 나눠라!!
x_train, x_test, y_train, y_test =  train_test_split(x, y, train_size=0.625, shuffle=True, random_state=66)
x_test, x_val, y_test, y_val =  train_test_split(x_test, y_test, test_size=0.5, shuffle=True, random_state=66)


# 2. 모델
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1,
          validation_data=(x_val, y_val))
# 1epoch마다 문제풀이시키기
# 통상적으로 train셋의 loss보다 안좋은 편 (데이터 사이즈자체가 작음 애초에)

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test) # 수능
print('loss : ', loss)

result = model.predict([17])
print('[17]의 예측값 : ', result)
