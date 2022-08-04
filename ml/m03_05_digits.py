from sklearn.datasets import load_digits
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


# 1. 데이터
datasets = load_digits()
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
y_predict = model.predict(x_test)
print('ypred: ', y_predict, '\n')

model = SVC()
model.fit(x_train, y_train)
result = model.score(x_test, y_test)
print('SVC acc 결과: ', result)
y_predict = model.predict(x_test)
print('ypred: ', y_predict, '\n')

model = Perceptron()
model.fit(x_train, y_train)
result = model.score(x_test, y_test)
print('Perceptron acc 결과: ', result)
y_predict = model.predict(x_test)
print('ypred: ', y_predict, '\n')

model = LogisticRegression()
model.fit(x_train, y_train)
result = model.score(x_test, y_test)
print('LogisticRegression acc 결과: ', result)
y_predict = model.predict(x_test)
print('ypred: ', y_predict, '\n')

model = KNeighborsClassifier()
model.fit(x_train, y_train)
result = model.score(x_test, y_test)
print('KNeighborsClassifier acc 결과: ', result)
y_predict = model.predict(x_test)
print('ypred: ', y_predict, '\n')

model = DecisionTreeClassifier()
model.fit(x_train, y_train)
result = model.score(x_test, y_test)
print('DecisionTreeClassifier acc 결과: ', result)
y_predict = model.predict(x_test)
print('ypred: ', y_predict, '\n')

model = RandomForestClassifier()
model.fit(x_train, y_train)
result = model.score(x_test, y_test)
print('RandomForestClassifier acc 결과: ', result)
y_predict = model.predict(x_test)
print('ypred: ', y_predict, '\n')

# LinearSVC acc 결과:  0.9611111111111111
# ypred:  [2 6 0 3 7 1 5 2 6 7 9 1 7 0 5 5 5 1 5 8 5 4 6 7 1 9 0 0 0 1 3 2 3 4 6 3 9
#  9 9 3 8 5 4 4 1 2 3 2 5 8 5 5 2 6 8 0 7 4 2 6 3 0 6 3 4 5 4 1 2 2 0 4 1 6
#  4 3 4 6 9 7 2 5 9 0 0 9 6 1 6 6 5 8 5 9 6 3 1 8 1 7 6 1 0 1 7 6 7 7 5 3 0
#  2 8 8 8 9 5 7 7 0 9 1 1 9 1 3 2 1 9 2 7 4 4 9 4 2 2 4 0 4 3 2 8 8 8 7 8 1
#  1 4 1 8 0 8 9 4 7 1 9 9 2 9 3 0 2 7 1 5 6 2 2 9 4 3 2 8 8 7 0 0]

# SVC acc 결과:  0.9888888888888889
# ypred:  [2 6 0 3 7 1 5 2 6 7 9 1 7 0 5 5 5 1 5 8 5 4 6 7 1 9 0 0 0 8 3 2 3 4 6 3 9
#  9 9 3 8 5 4 4 8 2 3 2 5 8 5 5 2 6 8 0 7 4 2 6 3 0 6 3 4 5 4 1 2 2 0 4 1 6
#  4 3 4 6 9 7 2 5 9 0 0 9 6 8 6 6 5 8 5 9 6 3 1 8 1 7 6 1 0 1 7 6 7 7 5 5 0
#  2 8 8 8 9 5 7 7 0 9 1 1 5 1 3 2 6 9 2 7 4 4 3 4 2 2 4 0 4 3 2 8 8 8 7 8 1
#  1 4 1 8 0 8 9 4 7 1 9 9 8 9 3 0 2 7 1 5 6 2 2 9 4 3 2 8 8 7 0 0]

# Perceptron acc 결과:  0.9388888888888889
# ypred:  [2 6 0 3 7 1 5 2 6 7 9 1 7 0 5 5 5 1 5 8 5 4 6 7 1 9 0 0 0 1 3 2 3 4 6 3 9
#  9 9 3 8 5 4 4 1 2 3 2 5 3 5 5 2 6 8 0 7 4 2 6 3 0 6 3 4 5 4 1 2 2 0 4 1 6
#  4 3 4 6 9 7 2 5 9 0 0 9 6 9 6 6 5 8 5 9 6 3 1 3 1 7 6 1 0 1 7 6 7 7 5 3 0
#  2 8 8 5 9 5 7 7 0 9 1 1 9 6 3 2 6 9 2 7 4 4 3 4 2 2 4 0 4 3 2 8 8 8 7 9 1
#  1 4 1 8 0 8 9 4 7 1 9 9 2 9 3 0 2 7 1 5 6 2 2 9 4 3 2 3 8 7 0 0]

# LogisticRegression acc 결과:  0.9611111111111111
# ypred:  [2 6 0 3 7 1 5 2 6 7 9 1 7 0 5 5 5 1 5 8 5 4 6 7 1 9 0 0 0 1 3 2 3 9 6 3 9
#  9 9 3 8 5 4 4 1 2 3 2 5 8 5 5 2 6 8 0 7 4 2 6 3 0 6 3 4 5 4 1 2 2 0 4 1 6
#  4 3 4 6 9 7 2 5 9 0 0 9 6 4 6 6 5 8 5 9 6 3 1 5 1 7 6 1 0 1 7 6 7 7 5 5 0
#  2 8 8 5 9 5 7 7 0 9 1 1 9 1 3 2 6 9 2 7 4 4 9 4 2 2 4 0 4 3 2 8 8 8 7 8 1
#  1 4 1 8 0 8 9 4 7 1 9 9 2 9 3 0 2 7 1 5 6 2 2 9 4 3 2 8 8 7 0 0]

# KNeighborsClassifier acc 결과:  0.9944444444444445
# ypred:  [2 6 0 3 7 1 5 2 6 7 9 1 7 0 5 5 5 1 5 8 5 4 6 7 1 9 0 0 0 8 3 2 3 4 6 3 9
#  9 9 3 8 5 4 4 8 2 3 2 5 8 5 5 2 6 8 0 7 4 2 6 3 0 6 3 4 5 4 1 2 2 0 4 1 6
#  4 3 4 6 9 7 2 5 9 0 0 9 6 4 6 6 5 8 5 9 6 3 1 8 1 7 6 1 0 1 7 6 7 7 5 5 0
#  2 8 8 8 9 5 7 7 0 9 1 1 5 1 3 2 6 9 2 7 4 4 3 4 2 2 4 0 4 3 2 8 8 8 7 8 1
#  1 4 1 8 0 8 9 4 7 1 9 9 8 9 3 0 2 7 1 5 6 2 2 9 4 3 2 8 8 7 0 0]

# DecisionTreeClassifier acc 결과:  0.8666666666666667
# ypred:  [2 6 0 3 7 1 5 2 6 7 3 1 7 0 5 5 9 6 5 8 5 4 6 7 1 9 0 0 0 1 3 2 3 8 6 3 1
#  9 9 3 8 5 4 4 8 2 7 2 5 1 5 5 2 6 2 0 7 4 2 6 3 0 6 3 4 5 4 1 2 2 0 4 1 6
#  4 3 1 6 9 7 2 5 8 0 0 8 6 4 2 6 5 8 5 9 6 3 1 8 1 7 6 1 0 1 7 6 7 7 5 3 0
#  2 8 8 3 9 4 7 7 0 9 1 3 3 1 3 2 4 9 8 8 1 4 3 4 2 2 4 0 4 3 2 8 8 8 7 8 1
#  1 4 1 8 0 8 9 4 7 1 9 9 8 3 3 0 2 7 1 5 6 2 2 9 4 3 2 8 2 7 0 0] 

# RandomForestClassifier acc 결과:  0.9833333333333333
# ypred:  [2 6 0 3 7 1 5 2 6 7 9 1 7 0 5 5 5 1 5 8 5 4 6 7 1 9 0 0 0 8 3 2 3 4 6 3 9
#  9 9 3 8 5 4 4 8 2 3 2 5 1 5 5 2 6 8 0 7 4 2 6 3 0 6 3 4 5 4 1 2 2 0 4 1 6
#  4 3 4 6 9 7 2 5 9 0 0 9 6 7 6 6 5 8 5 9 6 3 1 8 1 7 6 1 0 1 7 6 7 7 5 5 0
#  2 8 8 8 9 5 7 7 0 9 1 1 5 1 3 2 6 9 2 7 4 4 3 4 2 2 4 0 4 3 2 8 8 8 7 8 1
#  1 4 1 8 0 8 9 4 7 1 9 9 8 9 3 0 2 7 1 5 6 2 2 9 4 3 2 8 8 7 0 0]