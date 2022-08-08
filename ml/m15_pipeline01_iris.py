import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split, KFold
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=1234)

# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# 2. 모델
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import make_pipeline

model = make_pipeline(MinMaxScaler(), SVC()) # 굳이 변수명으로 정의하지 않아도 바로 갖다 쓸 수 있음

# 3. 훈련
model.fit(x_train, y_train) # pipeline의 fit에는 알아서 fit_transform이 들어가 있음

# 4. 평가, 예측
result = model.score(x_test, y_test) # pipeline의 score에는 알아서 transform이 들어가 있음

print('model.score: ', result)

# model.score:  1.0

# 스케일링 안했을때
# LinearSVC 결과:  1.0
# SVC 결과:  1.0
# Perceptron 결과:  1.0
# LogisticRegression 결과:  1.0
# KNeighborsClassifier 결과:  1.0
# DecisionTreeClassifier 결과:  1.0
# RandomForestClassifier 결과:  1.0