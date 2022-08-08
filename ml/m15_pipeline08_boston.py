import numpy as np
from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 1. 데이터
datasets = load_boston()
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

# model.score:  0.9172441370546218

# 스케일링 안했을 때
# LinearSVR r2 결과:  0.7434063515479603
# SVR r2 결과:  0.23474677555722312
# SVR r2 결과:  0.23474677555722312
# LinearRegression r2 결과:  0.8111288663608656
# KNeighborsRegressor r2 결과:  0.5900872726222293
# DecisionTreeRegressor r2 결과:  0.7780553674479604
# RandomForestRegressor r2 결과:  0.9204893478849648

# perceptron 오류남