import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split, KFold
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=1234)

# 2. 모델
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import make_pipeline, Pipeline

# model = make_pipeline(MinMaxScaler(), RandomForestClassifier())
model = Pipeline([('minmax', MinMaxScaler()), ('RF', RandomForestClassifier())])
# ('변수명', 스케일러())  <-이런식으로 씀

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
result = model.score(x_test, y_test)

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