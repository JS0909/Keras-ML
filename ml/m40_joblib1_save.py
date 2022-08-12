import numpy as np
from sklearn.datasets import load_breast_cancer, load_boston
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from xgboost import XGBClassifier, XGBRegressor
import time
from sklearn.metrics import accuracy_score, r2_score

# 1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=1234, train_size=0.8, stratify=y)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델
model = XGBClassifier(n_estimators=100,
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
              tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=0, random_state=1234,
              )

model.fit(x_train, y_train, early_stopping_rounds=10, 
          eval_set=[(x_train, y_train), (x_test, y_test)],
          eval_metric=['logloss'],
          )

print('테스트 스코어: ', model.score(x_test, y_test))

acc = accuracy_score(y_test, model.predict(x_test))
print('acc_score 결과: ', acc)


import joblib
path = 'D:/study_data/_save/_xg/'
joblib.dump(model, path + 'm40_joblib1_save.dat')

# 최상의 매개변수:  {'colsample_bylevel': 1, 'colsample_bynode': 1, 'colsample_bytree': 0.5, 'gamma': 0, 'learning_rate': 1, 'max_depth': 2, 'min_child_weight': 1, 'n_estimators': 100, 'reg_alpha': 0.01, 'reg_lambda': 1, 'subsample': 1}
# 최상의 점수:  0.9824175824175825
# 테스트 스코어:  0.9649122807017544

# n_estimators = 100
# 테스트 스코어:  0.9912280701754386
