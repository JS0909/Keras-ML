import numpy as np
from sklearn.datasets import load_iris

# 1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, train_size=0.8, random_state=1234)

# 2. 모델구성
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier # pip install xgboost

# model = DecisionTreeClassifier()
# model = RandomForestClassifier()
# model = GradientBoostingClassifier()
model = XGBClassifier()

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
result = model.score(x_test, y_test)
print('model.score: ', result)

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print('acc_score: ', acc)

print('----------------------------------------')
print(model, ': ', model.feature_importances_) # tree계열에만 있음

# DecisionTreeClassifier()     :  [0.         0.01669101 0.07659085 0.90671814] // 각 열별 중요도를 나타냄
# RandomForestClassifier()     :  [0.10766389 0.03133571 0.44818798 0.41281242]
# GradientBoostingClassifier() :  [0.00549151 0.01359283 0.30271053 0.67820512]
# XGBClassifier()              :  [0.00912187 0.0219429  0.678874   0.29006115]