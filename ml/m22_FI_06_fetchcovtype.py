import numpy as np
from sklearn.datasets import fetch_covtype

# 1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, train_size=0.8, random_state=1234)

# 2. 모델구성
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

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

models = [DecisionTreeClassifier(), RandomForestClassifier(), GradientBoostingClassifier(), XGBClassifier()]

# 3. 컴파일, 훈련, 평가, 예측
''' 훈련 + 그림
plt.figure(figsize=(10,5))
for i in range(len(models)):
    models[i].fit(x_train, y_train)
    plt.subplot(2,2, i+1)
    plot_feature_importances(models[i])
    if str(models[i]).startswith('XGBClassifier'):
        plt.title('XGB()')
    else:
        plt.title(models[i])
plt.show()
'''
for model in models:
    model.fit(x_train, y_train)
    score = model.score(x_test, y_test)
    if str(model).startswith('XGB'):
        print('XGB 의 스코어: ', score)
    else:
        print(str(model).strip('()'), '의 스코어: ', score)
    

