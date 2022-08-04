from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.datasets import fetch_covtype


# 1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target
x_train, x_test, y_train, y_test =  train_test_split(x, y, train_size=0.9, shuffle=True, random_state=86)


# 2. 모델구성
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron, LogisticRegression # !논리회귀(분류임)!
from sklearn.neighbors import KNeighborsClassifier # 최근접 이웃
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier # 결정트리를 여러개 랜덤으로 뽑아서 앙상블해서 봄

model = LinearSVC()
model.fit(x_train, y_train)
result = model.score(x_test, y_test)
print('LinearSVC acc 결과: ', result)
# y_predict = model.predict(x_test)
# print('ypred: ', y_predict, '\n')

model = SVC()
model.fit(x_train, y_train)
result = model.score(x_test, y_test)
print('SVC acc 결과: ', result)
# y_predict = model.predict(x_test)
# print('ypred: ', y_predict, '\n')

model = Perceptron()
model.fit(x_train, y_train)
result = model.score(x_test, y_test)
print('Perceptron acc 결과: ', result)
# y_predict = model.predict(x_test)
# print('ypred: ', y_predict, '\n')

model = LogisticRegression()
model.fit(x_train, y_train)
result = model.score(x_test, y_test)
print('LogisticRegression acc 결과: ', result)
# y_predict = model.predict(x_test)
# print('ypred: ', y_predict, '\n')

model = KNeighborsClassifier()
model.fit(x_train, y_train)
result = model.score(x_test, y_test)
print('KNeighborsClassifier acc 결과: ', result)
# y_predict = model.predict(x_test)
# print('ypred: ', y_predict, '\n')

model = DecisionTreeClassifier()
model.fit(x_train, y_train)
result = model.score(x_test, y_test)
print('DecisionTreeClassifier acc 결과: ', result)
# y_predict = model.predict(x_test)
# print('ypred: ', y_predict, '\n')

model = RandomForestClassifier()
model.fit(x_train, y_train)
result = model.score(x_test, y_test)
print('RandomForestClassifier acc 결과: ', result)
# y_predict = model.predict(x_test)
# print('ypred: ', y_predict, '\n')
