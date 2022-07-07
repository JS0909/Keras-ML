from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np

# 1. 데이터
x_train = np.array(range(1,11)) # 훈련용
y_train = np.array(range(1,11))
x_test = np.array([11,12,13]) # testset은 보통 evaluate 에서 사용 (수능)
y_test = np.array([11,12,13])
x_val = np.array([14,15,16]) # 문제풀어보기용
y_val = np.array([14,15,16])
# 총 데이터는 1~16

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
