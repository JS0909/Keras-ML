from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score,\
    RandomizedSearchCV
    
# 1. 데이터
datasets = load_boston()

x = datasets['data']
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=9)

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
# model = SVC(C=1, kernel='linear', degree=3)
model = RandomizedSearchCV(RandomForestRegressor(), parameters, cv=kfold, verbose=1, refit=True, n_jobs=-1)

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

# Fitting 5 folds for each of 10 candidates, totalling 50 fits
# 최적의 매개변수:  RandomForestRegressor(max_depth=8, n_estimators=200, n_jobs=-1)
# 최적의 파라미터:  {'n_jobs': -1, 'n_estimators': 200, 'max_depth': 8}
# best_score_:  0.8453217458178388     
# model.score:  0.888509650834215
# acc score:  0.888509650834215
# best tuned acc:  0.888509650834215
# 걸린시간:  3.59 초