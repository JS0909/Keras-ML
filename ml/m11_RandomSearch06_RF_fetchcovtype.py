from sklearn.datasets import fetch_covtype
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score,\
    RandomizedSearchCV

# 1. 데이터
datasets = fetch_covtype()

x = datasets['data']
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=9)

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=9)

parameters = [
    {'n_estimators':[100,200], 'max_depth':[6,8,10,12], 'n_jobs':[-1]},
    {'max_depth':[6,8,10,12], 'min_samples_leaf':[3,5,7,10], 'n_jobs':[-1]},
    {'min_samples_leaf':[3,5,7,10], 'min_samples_split':[2,3,5,10], 'n_jobs':[-1]},
    {'n_estimators':[100,200], 'max_depth':[6,8,10,12], 'min_samples_split':[2,3,5,10]},
    ]                                                                                       
                      
#2. 모델구성
from sklearn.ensemble import RandomForestClassifier
# model = SVC(C=1, kernel='linear', degree=3)
model = RandomizedSearchCV(RandomForestClassifier(), parameters, cv=kfold, verbose=1, refit=True, n_jobs=-1)

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
print('acc score: ', accuracy_score(y_test, ypred))
ypred_best = model.best_estimator_.predict(x_test)
print('best tuned acc: ', accuracy_score(y_test, ypred_best))

print('걸린시간: ', round(end-start,2), '초')

# Fitting 5 folds for each of 10 candidates, totalling 50 fits
# 최적의 매개변수:  RandomForestClassifier(min_samples_leaf=7, n_jobs=-1)
# 최적의 파라미터:  {'n_jobs': -1, 'min_samples_split': 2, 'min_samples_leaf': 7}
# best_score_:  0.9195260810415871
# model.score:  0.9273598788327324
# acc score:  0.9273598788327324
# best tuned acc:  0.9273598788327324
# 걸린시간:  762.35 초