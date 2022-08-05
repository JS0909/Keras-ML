from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score,\
    GridSearchCV # 격자 탐색, CV: cross validation

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


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8,shuffle=True, random_state=9)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=9)

parameters = [
    {'c':[1,10,100,1000], 'kernel':['linear'], 'degree':[3,4,5]},
    {'c':[1,10,100], 'kernel':['rbf'], 'gamma':[0.001, 0.0001]},
    {'c':[1,10,100,1000], 'kernel':['sigmoid'],
     'gamma':[0.01,0.001,0.0001], 'degree':[3,4]}
]
                      
#2. 모델구성
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron, LogisticRegression # !논리회귀(분류임)!
from sklearn.neighbors import KNeighborsClassifier # 최근접 이웃
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier # 결정트리를 여러개 랜덤으로 뽑아서 앙상블해서 봄

models = [LinearSVC, SVC, Perceptron, LogisticRegression, KNeighborsClassifier, DecisionTreeClassifier, RandomForestClassifier]

for model in models:
    model = model()
    model_name = str(model).strip('()')
    model.fit(x_train, y_train)
    result = model.score(x_test, y_test)
    print(model_name, '결과: ', result)
    
# LinearSVC 결과:  1.0
# SVC 결과:  1.0
# Perceptron 결과:  1.0
# LogisticRegression 결과:  1.0
# KNeighborsClassifier 결과:  1.0
# DecisionTreeClassifier 결과:  1.0
# RandomForestClassifier 결과:  1.0