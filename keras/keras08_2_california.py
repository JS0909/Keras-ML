from sklearn.datasets import fetch_california_housing
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


# 1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
x_train, x_test, y_train, y_test =  train_test_split(x, y, train_size=0.7, shuffle=True, random_state=66)

'''
print(x)
print(y)
print(x.shape) # (20640, 8)
print(y.shape) # (20640,)

print(datasets.feature_names)
# ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']

# print(datasets.DESCR)
'''

# [실습] 아래를 완성할 것
# R2 0.55~0.6


# 2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=8))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(50))
model.add(Dense(5))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam')
model.fit(x_train, y_train, epochs=500, batch_size=200)

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x)

r2 = r2_score(y, y_predict)
print('r2스코어 : ', r2)

# loss :  0.6562458276748657 
# r2스코어 :  0.5204211245083776