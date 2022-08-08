import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

# 1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets.target

allfeature = round(x.shape[1]*0.2, 0)
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

# 자를 갯수:  3
# DecisionTreeClassifier 의 스코어:         0.8888888888888888
# DecisionTreeClassifier 의 드랍후 스코어:  0.8888888888888888
# RandomForestClassifier 의 스코어:         0.9444444444444444
# RandomForestClassifier 의 드랍후 스코어:  0.9166666666666666
# GradientBoostingClassifier 의 스코어:         0.8611111111111112
# GradientBoostingClassifier 의 드랍후 스코어:  0.8611111111111112
# XGB 의 스코어:         0.8888888888888888
# XGB 의 드랍후 스코어:  0.8888888888888888