from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, KFold, cross_val_score,\
    GridSearchCV, HalvingRandomSearchCV
import warnings
warnings.filterwarnings('ignore')

# 1. 데이터
datasets = load_diabetes()

x = datasets['data']
y = datasets['target']

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
model = HalvingRandomSearchCV(RandomForestRegressor(), parameters, cv=kfold, verbose=1, refit=True, n_jobs=-1)

# 3. 컴파일, 훈련
import time
start = time.time()
model.fit(x, y)
end = time.time()

print('최적의 매개변수: ', model.best_estimator_)
print('최적의 파라미터: ', model.best_params_)
print('best_score_: ', model.best_score_)
print('model.score: ', model.score(x, y))
ypred = model.predict(x)
print('acc score: ', r2_score(y, ypred))
ypred_best = model.best_estimator_.predict(x)
print('best tuned acc: ', r2_score(y, ypred_best))

print('걸린시간: ', round(end-start,2), '초')

# ----------
# iter: 1
# n_candidates: 15
# n_resources: 30
# Fitting 5 folds for each of 15 candidates, totalling 75 fits
# ----------
# iter: 2
# n_candidates: 5
# n_resources: 90
# Fitting 5 folds for each of 5 candidates, totalling 25 fits
# ----------
# iter: 3
# n_candidates: 2
# n_resources: 270
# Fitting 5 folds for each of 2 candidates, totalling 10 fits
# 최적의 매개변수:  RandomForestRegressor(max_depth=12, min_samples_split=10, n_estimators=200)
# 최적의 파라미터:  {'n_estimators': 200, 'min_samples_split': 10, 'max_depth': 12}
# best_score_:  0.3792237943221957
# model.score:  0.8272497743392861
# acc score:  0.8272497743392861
# best tuned acc:  0.8272497743392861
# 걸린시간:  8.89 초