from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split


# 1. 데이터
x = np.array(range(1,17))
y = np.array(range(1,17))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=99)                        
# print(x_train.shape, x_test.shape) # (12,) (4,)

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
          validation_split=0.25) # train 데이터 내에서 25퍼센트를 가져와서 validation 데이터로 사용하겠다
# 보통 trainset:validationset:testset = 6:2:2 로 쪼갠다

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test) # 수능
print('loss : ', loss)

result = model.predict([17])
print('[17]의 예측값 : ', result)
