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
from sklearn.model_selection import train_test_split

# 1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target

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
    model_drop_cal = []
    model.fit(x_train, y_train)
    score = model.score(x_test, y_test)
    if str(model).startswith('XGB'):
        print('XGB 의 스코어: ', score)
    else:
        print(str(model).strip('()'), '의 스코어: ', score)
    for i in range(len(model.feature_importances_)):
        if model.feature_importances_[i]<=0.03:
            model_drop_cal.append(i)
    print('중요도낮은칼럼: ', model_drop_cal)
            
# DecisionTreeClassifier 의 스코어:  1.0
# 중요도낮은칼럼:  [0, 1]
# RandomForestClassifier 의 스코어:  1.0
# 중요도낮은칼럼:  [1]
# GradientBoostingClassifier 의 스코어:  1.0
# 중요도낮은칼럼:  [0, 1]
# XGB 의 스코어:  1.0
# 중요도낮은칼럼:  [0, 1]