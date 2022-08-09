import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target
le = LabelEncoder()

allfeature = round(x.shape[1]*0.2, 0)
print('자를 갯수: ', int(allfeature))


x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, train_size=0.8, random_state=1234)
y_train = le.fit_transform(y_train)

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
        
    x_af = np.delete(x, featurelist, axis=1)
    x_train2, x_test2, y_train2, y_test2 = train_test_split(x_af, y, shuffle=True, train_size=0.8, random_state=1234)
    y_train2 = le.fit_transform(y_train2)
    
    model.fit(x_train2, y_train2)
    score = model.score(x_test2, y_test2)
    if str(model).startswith('XGB'):
        print('XGB 의 드랍후 스코어: ', score)
    else:
        print(str(model).strip('()'), '의 드랍후 스코어: ', score)

# 자를 갯수:  11
# DecisionTreeClassifier 의 스코어:         0.9401048165710008
# DecisionTreeClassifier 의 드랍후 스코어:  0.9401822672392279
# RandomForestClassifier 의 스코어:         0.9565759920139756
# RandomForestClassifier 의 드랍후 스코어:  0.957333287436641
# GradientBoostingClassifier 의 스코어:         0.7737579924786796
# GradientBoostingClassifier 의 드랍후 스코어:  0.7737493868488765
# XGB 의 스코어:         0.05807078991075962
# XGB 의 드랍후 스코어:  0.057038114334397566

# y라벨 오류는 라벨인코더로 해결하면 됨, XGB에서 칼럼 오류남
# 아니면 train_test_split에서 stratify=y 해주면 해결됨