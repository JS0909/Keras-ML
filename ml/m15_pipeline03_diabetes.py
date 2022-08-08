import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split, KFold
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=1234)

# 2. 모델
from sklearn.svm import LinearSVR, SVR
from sklearn.ensemble import RandomForestRegressor

from sklearn.pipeline import make_pipeline

model = make_pipeline(MinMaxScaler(), RandomForestRegressor()) # 굳이 변수명으로 정의하지 않아도 바로 갖다 쓸 수 있음

# 3. 훈련
model.fit(x_train, y_train) # pipeline의 fit에는 알아서 fit_transform이 들어가 있음

# 4. 평가, 예측
result = model.score(x_test, y_test) # pipeline의 score에는 알아서 transform이 들어가 있음

print('model.score: ', result)

# model.score:  0.40905093710413065

# 스케일링 안했을 때
# LinearSVR 결과:  -0.21026793010486222
# SVR 결과:  0.2668794571758186
# Perceptron 결과:  0.0
# LinearRegression 결과:  0.6557534150889773
# KNeighborsRegressor 결과:  0.5704639112420011
# DecisionTreeRegressor 결과:  -0.054909125498723066
# RandomForestRegressor 결과:  0.6329634639445246