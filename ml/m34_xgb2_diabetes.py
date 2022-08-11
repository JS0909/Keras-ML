import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from xgboost import XGBClassifier, XGBRegressor
import time

# 1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
# print(x.shape, y.shape) # (569, 30) (569,)

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=1234, train_size=0.8)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1234)

# 'n_estimators':[100,200,300,400,500,1000] // 디폴트 100 / 1~inf 정수
# 'learning_rate' 혹은 'eta':[0.1,0.2,0.3,0.5,1,0.01,0.001] // 디폴트 0.3 / 0~1
# 'max_depth':[None,2,3,4,5,6,7,8,9,10] // 디폴트 6 / 0~inf 정수 / max면 낮게 잡을 수록 좋은 편 - 과적합될 수 있기 때문에, 통상 4로 잡으면 좋음, None은 무한대
# 'gamma':[0,1,2,3,4,5,7,10,100] // 디폴트 0 / 0~inf
# 'min_child_weight':[0,0.1,0.001,0.5,1,5,10,100] // 디폴트 1 / 0~inf
# 'subsample':[0,0.1,0.2,0.3,0.5,0.7,1] // 디폴트 1 / 0~1
# 'colsample_bytree':[0,0.1,0.2,0.3,0.5,0.7,1] // 디폴트 1 / 0~1
# 'colsample_bylevel':[0,0.1,0.2,0.3,0.5,0.7,1] // 디폴트 1 / 0~1
# 'colsample_bynode':[0,0.1,0.2,0.3,0.5,0.7,1] // 디폴트 1 / 0~1
# 'reg_alpha' 혹은 'alpha':[0,0.1,0.01,0.001,1,2,10] // 디폴트 0 / 0~inf / L1 절대값 가중치 규제
# 'reg_lambda' 혹은 'lambda':[0,0.1,0.01,0.001,1,2,10] // 디폴트 1 / 0~inf / L2 제곱 가중치 규제



parameters = {'n_estimators':[200],
              'learning_rate':[0.1],
              'max_depth':[None],
              'gamma':[4],
              'min_child_weight':[0],
              'subsample':[1],
              'colsample_bytree':[1],
              'colsample_bylevel':[0.2],
              'colsample_bynode':[1],
              'reg_alpha':[0.1],
              'reg_lambda':[0.1]
              } 

# 2. 모델
xgb = XGBRegressor(random_state=1234)
model = GridSearchCV(xgb, parameters, cv=kfold, n_jobs=-1)

model.fit(x_train, y_train)

print('최상의 매개변수: ', model.best_params_)
print('최상의 점수: ', model.best_score_)
print('테스트 스코어: ', model.score(x_test, y_test))

# 최상의 매개변수:  {'n_estimators': 200}
# 최상의 점수:  0.8653618157471552
# 테스트 스코어:  0.9110739449829566

# 최상의 매개변수:  {'learning_rate': 0.1, 'n_estimators': 200}
# 최상의 점수:  0.870406914722434
# 테스트 스코어:  0.9190478319847336

# 최상의 매개변수:  {'learning_rate': 0.1, 'max_depth': None, 'n_estimators': 200}
# 최상의 점수:  0.870406914722434
# 테스트 스코어:  0.9190478319847336

# 최상의 매개변수:  {'gamma': 4, 'learning_rate': 0.1, 'max_depth': None, 'n_estimators': 200}
# 최상의 점수:  0.871947485836839
# 테스트 스코어:  0.911374230458004

# 최상의 매개변수:  {'gamma': 4, 'learning_rate': 0.1, 'max_depth': None, 'min_child_weight': 0, 'n_estimators': 200}
# 최상의 점수:  0.871947485836839
# 테스트 스코어:  0.911374230458004

# 최상의 매개변수:  {'gamma': 4, 'learning_rate': 0.1, 'max_depth': None, 'min_child_weight': 0, 'n_estimators': 200, 'subsample': 1}
# 최상의 점수:  0.871947485836839
# 테스트 스코어:  0.911374230458004

# 최상의 매개변수:  {'colsample_bytree': 1, 'gamma': 4, 'learning_rate': 0.1, 'max_depth': None, 'min_child_weight': 0, 'n_estimators': 200, 'subsample': 1}
# 최상의 점수:  0.871947485836839
# 테스트 스코어:  0.911374230458004

# 최상의 매개변수:  {'colsample_bylevel': 0.2, 'colsample_bytree': 1, 'gamma': 4, 'learning_rate': 0.1, 'max_depth': None, 'min_child_weight': 0, 'n_estimators': 200, 'subsample': 1}
# 최상의 점수:  0.8730617754711222
# 테스트 스코어:  0.9031864829196466

# 최상의 매개변수:  {'colsample_bylevel': 0.2, 'colsample_bynode': 1, 'colsample_bytree': 1, 'gamma': 4, 'learning_rate': 0.1, 'max_depth': None, 'min_child_weight': 0, 'n_estimators': 200, 'subsample': 1}
# 최상의 점수:  0.8730617754711222
# 테스트 스코어:  0.9031864829196466

# 최상의 매개변수:  {'colsample_bylevel': 0.2, 'colsample_bynode': 1, 'colsample_bytree': 1, 'gamma': 4, 'learning_rate': 0.1, 'max_depth': None, 'min_child_weight': 0, 'n_estimators': 200, 'reg_alpha': 0.01, 'subsample': 1}
# 최상의 점수:  0.8761055335318225
# 테스트 스코어:  0.8982350219676197

# 최상의 매개변수:  {'colsample_bylevel': 0.2, 'colsample_bynode': 1, 'colsample_bytree': 1, 'gamma': 4, 'learning_rate': 0.1, 'max_depth': None, 'min_child_weight': 0, 'n_estimators': 200, 'reg_alpha': 0.1, 'reg_lambda': 0.1, 'subsample': 1}
# 최상의 점수:  0.874409331610881
# 테스트 스코어:  0.8897857954106936