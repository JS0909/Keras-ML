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

# 2. 모델구성
from sklearn.svm import LinearSVR, SVR
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

models = [LinearSVR, SVR, Perceptron, LinearRegression, KNeighborsRegressor, DecisionTreeRegressor, RandomForestRegressor]

for model in models:
    model = model()
    model_name = str(model).strip('()')
    model.fit(x_train, y_train)
    result = model.score(x_test, y_test)
    print(model_name, '결과: ', result)
    
# LinearSVR r2 결과:  -6.324623991048357
# SVR r2 결과:  -0.0370628287403556
# LinearRegression r2 결과:  0.6001949284390904
# KNeighborsRegressor r2 결과:  0.13188588139050017
# DecisionTreeRegressor r2 결과:  0.5967622571521326
# RandomForestRegressor r2 결과:  0.8091725373390208