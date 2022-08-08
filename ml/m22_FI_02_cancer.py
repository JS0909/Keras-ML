import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# 1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

allfeature = round(x.shape[1]*0.2, 1)
print('자를 갯수: ', int(allfeature))


x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, train_size=0.8, random_state=1234)

# 2. 모델구성
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

models = [DecisionTreeClassifier(), RandomForestClassifier(), GradientBoostingClassifier(), XGBClassifier()]

# 3. 컴파일, 훈련, 평가, 예측
for model in models:
    model.fit(x_train, y_train)
    score = model.score(x_test, y_test)
    if str(model).startswith('XGB'):
        print('XGB 의 스코어: ', score)
    else:
        print(str(model).strip('()'), '의 스코어: ', score)
    for a in range(int(allfeature)):
        print(np.argsort(model.feature_importances_)[a])

# 자를 갯수:  6.0
# DecisionTreeClassifier 의 스코어:  0.8947368421052632
# 중요도낮은칼럼순서:  [ 0 20 19 18 16 15 28 12 11 14  2  1  9  3  4 29  6  7  8  5 25 22 10 26
#  21 13 17 24 23 27]
# RandomForestClassifier 의 스코어:  0.9298245614035088
# 중요도낮은칼럼순서:  [11 14  9 18  8 17 29 19 16 28  5 15  4  1 12 25 24 10 21 26 13  3  2  0
#   7  6 23 20 22 27]
# GradientBoostingClassifier 의 스코어:  0.9122807017543859
# 중요도낮은칼럼순서:  [17  9 14 11  8  0 15 16 25 19  2 18  5 28  6 12  4 26  3 29 10 13  1 24
#  21 20 22 23  7 27]
# XGB 의 스코어:  0.9385964912280702
# 중요도낮은칼럼순서:  [ 8 12 11 19  5 17  2 10 16  9 28 25 14 15  4 18 26 24  0  1  6 13 21  3
#  29 22 23 20 27  7]



    

datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, train_size=0.8, random_state=1234)
