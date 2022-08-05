from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, KFold,\
    HalvingGridSearchCV


import sklearn as sk
print(sk.__version__) # 0.24.2

# 1. 데이터
datasets = load_boston()

x = datasets['data']
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.95, shuffle=True, random_state=9)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=9)

parameters = [
    {'n_estimators':[100,200], 'max_depth':[6,8,10,12], 'n_jobs':[-1,2,4]},
    {'max_depth':[6,8,10,12], 'min_samples_leaf':[3,5,7,10], 'n_jobs':[-1,2,4]},
    {'min_samples_leaf':[3,5,7,10], 'min_samples_split':[2,3,5,10], 'n_jobs':[-1,2,4]},
    {'n_estimators':[100,200], 'max_depth':[6,8,10,12], 'min_samples_split':[2,3,5,10]},
    ]                                                                                       
                      
#2. 모델구성
from sklearn.ensemble import RandomForestRegressor
model = HalvingGridSearchCV(RandomForestRegressor(), parameters, cv=kfold, verbose=1, refit=True, n_jobs=-1)

# 3. 컴파일, 훈련
import time
start = time.time()
model.fit(x_train, y_train)
end = time.time()

print('최적의 매개변수: ', model.best_estimator_)
print('최적의 파라미터: ', model.best_params_)
print('best_score_: ', model.best_score_)
print('model.score: ', model.score(x_test, y_test))
ypred = model.predict(x_test)
print('acc score: ', r2_score(y_test, ypred))
ypred_best = model.best_estimator_.predict(x_test)
print('best tuned acc: ', r2_score(y_test, ypred_best))

print('걸린시간: ', round(end-start,2), '초')

# n_iterations: 4
# n_required_iterations: 5
# n_possible_iterations: 4
# min_resources_: 10
# max_resources_: 480
# aggressive_elimination: False
# factor: 3
# ----------
# iter: 0
# n_candidates: 152
# n_resources: 10
# Fitting 5 folds for each of 152 candidates, totalling 760 fits
# ----------
# iter: 1
# n_candidates: 51
# n_resources: 30
# Fitting 5 folds for each of 51 candidates, totalling 255 fits
# ----------
# iter: 2
# n_candidates: 17
# n_resources: 90
# Fitting 5 folds for each of 17 candidates, totalling 85 fits
# ----------
# iter: 3
# n_candidates: 6
# n_resources: 270
# Fitting 5 folds for each of 6 candidates, totalling 30 fits
# 최적의 매개변수:  RandomForestRegressor(max_depth=12, min_samples_split=3, n_estimators=200)
# 최적의 파라미터:  {'max_depth': 12, 'min_samples_split': 3, 'n_estimators': 200}
# best_score_:  0.7916870315392487
# model.score:  0.9391516416955898
# acc score:  0.9391516416955898
# best tuned acc:  0.9391516416955898
# 걸린시간:  25.04 초