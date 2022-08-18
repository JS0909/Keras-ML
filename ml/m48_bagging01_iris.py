import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE


# 1. 데이터
datasets = load_iris()
x, y = datasets.data, datasets.target

allfeature = round(x.shape[1]*0.2, 0)
print('자를 갯수: ', int(allfeature))

print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1234, shuffle=True, stratify=y)
scl = MinMaxScaler()
x_train = scl.fit_transform(x_train)
x_test = scl.transform(x_test)

smote = SMOTE(random_state=1234)
x_train, y_train = smote.fit_resample(x_train, y_train)

# 2. 모델
from sklearn.ensemble import BaggingClassifier
from xgboost import XGBClassifier

xgb = XGBClassifier(tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=0, random_state=1234)
model = BaggingClassifier(xgb, n_estimators=100, n_jobs=-1, random_state=1234)

xgb.fit(x_train, y_train)
print(xgb.feature_importances_)

xaf = np.delete(x, np.argsort(xgb.feature_importances_)[0], axis=1)
x_train, x_test, y_train, y_test = train_test_split(xaf, y, train_size=0.8, random_state=1234, shuffle=True, stratify=y)
scl = MinMaxScaler()
x_train = scl.fit_transform(x_train)
x_test = scl.transform(x_test)


# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
print(model.score(x_test, y_test))

# 0.9333333333333333
