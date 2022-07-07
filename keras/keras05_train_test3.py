import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

# [검색] train과 test를 섞어서 7:3으로 찾을 수 있는 방법을 찾아라.
# 힌트: 사이킷런

x_train, x_test, y_train, y_test =  train_test_split(x, y, 
                                                     train_size=0.7, 
                                                      shuffle=True, # 명시 안할 경우 True가 default
                                                     random_state=1004)
# seed : 랜덤 난수에 따라 정해진 랜덤치로 뽑음 난수 바뀌기 전까지 고정되는 셈
# size : 퍼센트로 접근

print(x_test) # [ 1  9 10]
print(x_train) # [2 7 6 3 4 8 5]
print(y_test)
print(y_train)

# 디버그 시 줄 옆에 빨간 점 + F5
# 빨간 점 전까지 실행, 다시 F5 누르면 거기서부터 이어서 실행, 빈 곳 말고 내용 있는 곳 찍어야됨

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

# model.add(Dense(10, input_dim=1))
# model.add(Dense(1))
# epochs=100, batch_size=1
# loss :  7.579122740649855e-14
# [11]의 예측값 :  [[11.]]