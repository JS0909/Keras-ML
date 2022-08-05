from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score,\
    GridSearchCV, RandomizedSearchCV # 그리드 서치로 파라미터 가져오는 것 중, 랜덤으로 적용해봄

# 1. 데이터
datasets = load_iris()
print(datasets.DESCR)
'''
- class:
                - Iris-Setosa
                - Iris-Versicolour
                - Iris-Virginica
y값이 3개
이 3개 꽃 중 하나가 나와야 함
3중 분류
'''
print(datasets.feature_names)
x = datasets['data']
y = datasets['target']
print(x, '\n', y)
print(x.shape) # (150, 4)
print(y.shape) # (150,)
print('y의 라벨값: ', np.unique(y))


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=9)

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=9)

parameters = [
    {'C':[1,10,100,1000], 'kernel':['linear'], 'degree':[3,4,5]},                                # 12
    {'C':[1,10,100], 'kernel':['rbf'], 'gamma':[0.001, 0.0001]},                                 # 6
    {'C':[1,10,100,1000], 'kernel':['sigmoid'], 'gamma':[0.01,0.001,0.0001], 'degree':[3,4]}     # 24
]                                                                                                # 총 42회 파라미터 해봄
                      
#2. 모델구성
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron, LogisticRegression # !논리회귀(분류임)!
from sklearn.neighbors import KNeighborsClassifier # 최근접 이웃
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier # 결정트리를 여러개 랜덤으로 뽑아서 앙상블해서 봄

# model = RandomizedSearchCV(C=1, kernel='linear', degree=3)
model = RandomizedSearchCV(SVC(), parameters, cv=kfold, verbose=1, refit=True, n_jobs=-1, n_iter=15) # n_iter=10 디폴트, 파라미터 조합 중 열가지만 해본다
# refit: True면 최적의 파라미터로 훈련, False면 해당 파라미터로 훈련하지 않고 마지막 파라미터로 훈련
# n_jobs: cpu의 갯수를 몇개 사용할것인지

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
# 최적의 매개변수:  SVC(C=10, kernel='linear')
# 최적의 파라미터:  {'kernel': 'linear', 'degree': 3, 'C': 10}
# best_score_:  0.9666666666666666
# model.score:  1.0
# acc score:  1.0
# best tuned acc:  1.0
# 걸린시간:  1.8 초