import numpy as np
import pandas as pd

from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. 데이터
datasets = load_wine()
x, y = datasets.data, datasets.target

print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1234, shuffle=True, stratify=y)
scl = MinMaxScaler()
x_train = scl.fit_transform(x_train)
x_test = scl.transform(x_test)


# 2. 모델
from sklearn.ensemble import BaggingClassifier
from xgboost import XGBClassifier

model = BaggingClassifier(XGBClassifier(n_estimators=100,
              learning_rate=1,
              max_depth=2,
              gamma=0,
              min_child_weight=1,
              subsample=1,
              colsample_bytree=0.5,
              colsample_bylevel=1,
              colsample_bynode=1,
              reg_alpha=0.01,
              reg_lambd=1,
              tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=0, random_state=1234,), n_estimators=100, n_jobs=-1, random_state=1234)

# 3. 훈련
model.fit(x_train, y_train)


# 4. 평가, 예측
print(model.score(x_test, y_test))

# 1.0