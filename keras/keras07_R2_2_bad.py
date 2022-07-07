# 1. R2를 음수가 아닌 0.5 이하로 만들 것
# 2. 데이터 건들지 마
# 3. 레이어는 인풋 아웃풋 포함 7개 이상
# 4. batch_size = 1
# 5. 히든레이어의 노드는 10개 이상 100개 이하
# 6. train 70%
# 7. epoch 100번 이상
# 8. loss지표는 mse, mae
# [실습 시작]

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

# 1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,3,4,5,7,9,3,8,12,13,8,14,15,9,6,17,23,21,20])

# train_test_split
x_train, x_test, y_train, y_test =  train_test_split(x, y, train_size=0.7, shuffle=True, random_state=66)


# 2. 모델구성
model = Sequential()
model.add(Dense(15, input_dim=1))
model.add(Dense(81))
model.add(Dense(81))
model.add(Dense(81))
model.add(Dense(81))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(22))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(11))
model.add(Dense(70))
model.add(Dense(70))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss = 'mae', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)


# 4. 평가, 예측 
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x)

# R2 스코어, 결정계수 : 정확도가 대충 몇프로 나온다 라고 생각하면 됨
# 메트릭스엔 보통 평가지표 많이 나옴
from sklearn.metrics import r2_score
r2 = r2_score(y, y_predict)
print('r2스코어 : ', r2)


# model.add(Dense(11, input_dim=1))
# model.add(Dense(10))
# model.add(Dense(11))
# model.add(Dense(11))
# model.add(Dense(11))
# model.add(Dense(100))
# model.add(Dense(50))
# model.add(Dense(11))
# model.add(Dense(15))
# model.add(Dense(100))
# model.add(Dense(10))
# model.add(Dense(11))
# model.add(Dense(11))
# model.add(Dense(1))
# loss :  1.4122114181518555
# r2스코어 :  0.41402253928031996


# model.add(Dense(15, input_dim=1))
# model.add(Dense(81))
# model.add(Dense(81))
# model.add(Dense(81))
# model.add(Dense(81))
# model.add(Dense(10))
# model.add(Dense(10))
# model.add(Dense(100))
# model.add(Dense(22))
# model.add(Dense(10))
# model.add(Dense(10))
# model.add(Dense(11))
# model.add(Dense(70))
# model.add(Dense(70))
# model.add(Dense(1))
# loss :  3.7385571002960205
# r2스코어 :  0.09563123245986449


####################################################################
# loss :  1.4848469495773315
# r2스코어 :  0.7384583715657111
####################################################################

# import matplotlib.pyplot as plt 

# plt.scatter(x, y,color='green')
# plt.plot(x, y_predict, color='blue')
# plt.show()
