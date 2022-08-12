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

# 2. 모델, 3. 훈련
model = XGBClassifier()

# model.fit(x_train, y_train, early_stopping_rounds=10, 
#           eval_set=[(x_train, y_train), (x_test, y_test)],
#           eval_metric=['logloss'],
#           )

path = 'D:/study_data/_save/_xg/'
model.load_model(path+'m41_xgb1_save_model.dat')

# 4. 평가, 예측
print('테스트 스코어: ', model.score(x_test, y_test))

acc = accuracy_score(y_test, model.predict(x_test))
print('acc_score 결과: ', acc)


# 최상의 매개변수:  {'colsample_bylevel': 1, 'colsample_bynode': 1, 'colsample_bytree': 0.5, 'gamma': 0, 'learning_rate': 1, 'max_depth': 2, 'min_child_weight': 1, 'n_estimators': 100, 'reg_alpha': 0.01, 'reg_lambda': 1, 'subsample': 1}
# 최상의 점수:  0.9824175824175825
# 테스트 스코어:  0.9649122807017544

# n_estimators = 100
# 테스트 스코어:  0.9912280701754386
