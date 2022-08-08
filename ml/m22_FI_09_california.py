import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# 1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

allfeature = round(x.shape[1]*0.2, 0)
print('자를 갯수: ', int(allfeature))


x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, train_size=0.8, random_state=1234)

# 2. 모델구성
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

models = [DecisionTreeRegressor(), RandomForestRegressor(), GradientBoostingRegressor(), XGBRegressor()]

# 3. 컴파일, 훈련, 평가, 예측
for model in models:
    model.fit(x_train, y_train)
    score = model.score(x_test, y_test)
    if str(model).startswith('XGB'):
        print('XGB 의 스코어:        ', score)
    else:
        print(str(model).strip('()'), '의 스코어:        ', score)
        
    featurelist = []
    for a in range(int(allfeature)):
        featurelist.append(np.argsort(model.feature_importances_)[a])
        
    x_bf = np.delete(x, featurelist, axis=1)
    x_train2, x_test2, y_train2, y_test2 = train_test_split(x_bf, y, shuffle=True, train_size=0.8, random_state=1234)
    model.fit(x_train2, y_train2)
    score = model.score(x_test2, y_test2)
    if str(model).startswith('XGB'):
        print('XGB 의 드랍후 스코어: ', score)
    else:
        print(str(model).strip('()'), '의 드랍후 스코어: ', score)

# 자를 갯수:  2
# DecisionTreeRegressor 의 스코어:         0.5979213749039181
# DecisionTreeRegressor 의 드랍후 스코어:  0.6117278388953841
# RandomForestRegressor 의 스코어:         0.8061822867356276
# RandomForestRegressor 의 드랍후 스코어:  0.8058750624175282
# GradientBoostingRegressor 의 스코어:         0.7868072463466687
# GradientBoostingRegressor 의 드랍후 스코어:  0.7833602753783532
# XGB 의 스코어:         0.8266362514646761
# XGB 의 드랍후 스코어:  0.8307509185446069