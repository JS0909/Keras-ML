from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, StratifiedKFold,\
    HalvingRandomSearchCV

# 1. 데이터
datasets = load_iris()

x = datasets['data']
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=9)

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=9)

parameters = [
    {'n_estimators':[100,200], 'max_depth':[6,8,10,12], 'n_jobs':[-1,2,4]},
    {'max_depth':[6,8,10,12], 'min_samples_leaf':[3,5,7,10], 'n_jobs':[-1,2,4]},
    {'min_samples_leaf':[3,5,7,10], 'min_samples_split':[2,3,5,10], 'n_jobs':[-1,2,4]},
    {'n_estimators':[100,200], 'max_depth':[6,8,10,12], 'min_samples_split':[2,3,5,10]},
    ]                                                                                       
                      
#2. 모델구성
from sklearn.ensemble import RandomForestClassifier
model = HalvingRandomSearchCV(RandomForestClassifier(), parameters, cv=kfold, verbose=1, refit=True, n_jobs=-1, aggressive_elimination=True, factor=2)
# aggressive_elimination=True // 워닝 무시
# factor=10 (디폴트) // min_resources로 처음에 빼서 사용하고 다음회차부터는 min_resources*factor로 n_resources값을 정해줌
# 만약 max_resources가 factor*min_resources, 즉 다음회차의 n_resources보다 적어져버리게 된다면 min_resources로만 한번 돌리고 만다
# 만약 max_resources >= factor*min_resources 이고 그다음 회차도 그렇다면 max_resouces <= factor*min_resources가 될때까지 반복수행한다

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

# n_iterations: 2
# n_required_iterations: 2
# n_possible_iterations: 2
# min_resources_: 30
# max_resources_: 120
# aggressive_elimination: False
# factor: 3
# ----------
# iter: 0
# n_candidates: 4
# n_resources: 30
# Fitting 5 folds for each of 4 candidates, totalling 20 fits
# ----------
# iter: 1
# n_candidates: 2
# n_resources: 90
# Fitting 5 folds for each of 2 candidates, totalling 10 fits
# 최적의 매개변수:  RandomForestClassifier(min_samples_leaf=3, min_samples_split=3, n_jobs=2)
# 최적의 파라미터:  {'n_jobs': 2, 'min_samples_split': 3, 'min_samples_leaf': 3}
# best_score_:  0.9333333333333333
# model.score:  1.0
# acc score:  1.0
# best tuned acc:  1.0
# 걸린시간:  2.81 초