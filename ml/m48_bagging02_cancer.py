import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

# 1. 데이터
datasets = load_breast_cancer()
x, y = datasets.data, datasets.target

print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1234, shuffle=True, stratify=y)
scl = MinMaxScaler()
x_train = scl.fit_transform(x_train)
x_test = scl.transform(x_test)

parameters = {'n_estimators':[100],
              'learning_rate':[1],
              'max_depth':[2],
              'gamma':[0],
              'min_child_weight':[1],
              'subsample':[1],
              'colsample_bytree':[0.5],
              'colsample_bylevel':[1],
              'colsample_bynode':[1],
              'reg_alpha':[0.01],
              'reg_lambda':[0,0.1,0.01,0.001,1,2,10]
              } 

# 2. 모델
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# xgb = XGBClassifier(random_state=1234)
# gs = GridSearchCV(xgb, parameters, cv=5, n_jobs=-1)
# model = BaggingClassifier(gs, n_estimators=100, n_jobs=-1, random_state=1234)

dt = DecisionTreeClassifier(random_state=1234)
model = BaggingClassifier(dt, n_estimators=100, n_jobs=-1, random_state=1234)


# 3. 훈련
model.fit(x_train, y_train)


# 4. 평가, 예측
print(model.score(x_test, y_test))

# 0.9649122807017544
