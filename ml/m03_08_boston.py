from sklearn.datasets import load_boston
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time

# 1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target
x_train, x_test, y_train, y_test =  train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)


#2. 모델구성
from sklearn.svm import LinearSVR, SVR
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

model = LinearSVR()
model.fit(x_train, y_train)
result = model.score(x_test, y_test)
print('LinearSVR r2 결과: ', result)
# y_predict = model.predict(x_test)
# print('ypred: ', y_predict, '\n')

model = SVR()
model.fit(x_train, y_train)
result = model.score(x_test, y_test)
print('SVR r2 결과: ', result)
# y_predict = model.predict(x_test)
# print('ypred: ', y_predict, '\n')

# model = Perceptron()
# model.fit(x_train, y_train)
# result = model.score(x_test, y_test)
# print('Perceptron r2 결과: ', result)
# # y_predict = model.predict(x_test)
# # print('ypred: ', y_predict, '\n')

model = LinearRegression()
model.fit(x_train, y_train)
result = model.score(x_test, y_test)
print('LinearRegression r2 결과: ', result)
# y_predict = model.predict(x_test)
# print('ypred: ', y_predict, '\n')

model = KNeighborsRegressor()
model.fit(x_train, y_train)
result = model.score(x_test, y_test)
print('KNeighborsRegressor r2 결과: ', result)
# y_predict = model.predict(x_test)
# print('ypred: ', y_predict, '\n')

model = DecisionTreeRegressor()
model.fit(x_train, y_train)
result = model.score(x_test, y_test)
print('DecisionTreeRegressor r2 결과: ', result)
# y_predict = model.predict(x_test)
# print('ypred: ', y_predict, '\n')

model = RandomForestRegressor()
model.fit(x_train, y_train)
result = model.score(x_test, y_test)
print('RandomForestRegressor r2 결과: ', result)
# y_predict = model.predict(x_test)
# print('ypred: ', y_predict, '\n')

# LinearSVR r2 결과:  0.7434063515479603
# SVR r2 결과:  0.23474677555722312
# SVR r2 결과:  0.23474677555722312
# LinearRegression r2 결과:  0.8111288663608656
# KNeighborsRegressor r2 결과:  0.5900872726222293
# DecisionTreeRegressor r2 결과:  0.7780553674479604
# RandomForestRegressor r2 결과:  0.9204893478849648

# perceptron 오류남