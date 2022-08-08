# 실습
# 피처임포턴스가 전체 중요도에서 하위 20~25% 칼럼들을 제거하여
# 데이터셋 재구성 후
# 각 모델별로 돌려서 결과 도출

# 기존 모델결과와 비교

# 결과비교
# 1. DecisionTree
# 기존 acc: 
# 칼럼삭제 후 acc:

import numpy as np
from sklearn.datasets import load_iris

# 1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, train_size=0.8, random_state=1234)

# 2. 모델구성
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

'''
import matplotlib.pyplot as plt

def plot_feature_importances(model): # 그림 함수 정의
    n_features = datasets.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
                # x                     y
    plt.yticks(np.arange(n_features), datasets.feature_names) # 눈금 설정
    plt.xlabel('Feature Importances')
    plt.ylabel('Features')
    plt.ylim(-1, n_features) # ylimit : 축의 한계치 설정
'''

models = [DecisionTreeRegressor(), RandomForestRegressor(), GradientBoostingRegressor(), XGBRegressor()]

# 3. 컴파일, 훈련, 평가, 예측
''' 훈련 + 그림
plt.figure(figsize=(10,5))
for i in range(len(models)):
    models[i].fit(x_train, y_train)
    plt.subplot(2,2, i+1)
    plot_feature_importances(models[i])
    if str(models[i]).startswith('XGBRegressor'):
        plt.title('XGB()')
    else:
        plt.title(models[i])
plt.show()
'''
for model in models:
    model.fit(x_train, y_train)
    score = model.score(x_test, y_test)
    if str(model).startswith('XGB'):
        print('XGB의 스코어: ', score)
    else:
        print(str(model).strip('()'), '의 스코어: ', score)
    

# DecisionTreeRegressor 의 스코어:  1.0
# RandomForestRegressor 의 스코어:  0.9998703339882122
# GradientBoostingRegressor 의 스코어:  0.9983744292842164
# XGB 의 스코어:  0.9998069257068463