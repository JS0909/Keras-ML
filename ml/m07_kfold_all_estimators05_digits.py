from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore') # warnig 출력 안함

# 1. 데이터
datasets = load_digits()
print(datasets.DESCR)
print(datasets.feature_names)
x = datasets['data']
y = datasets['target']
print(x, '\n', y)
print(x.shape) # (150, 4)
print(y.shape) # (150,)
print('y의 라벨값: ', np.unique(y))


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8,shuffle=True, random_state=9)
         
from sklearn.preprocessing import MinMaxScaler
scaler= MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=99)
         
#2. 모델구성
allAlgorithms = all_estimators(type_filter='classifier')
print('allAlgorithms: ', allAlgorithms)
print('모델의 개수: ', len(allAlgorithms)) # 41

for (name, algorithm) in allAlgorithms:
    try:
        model = algorithm()
        ypred = cross_val_predict(model, x_test, y_test, cv=kfold)
        score = cross_val_score(model, x_train, y_train, cv=kfold)
        print(name, '의 정답률: ', round(np.mean(score),4))
        
    except:
        # continue # 또는 pass
        print(name, '은 안나온 놈')

# AdaBoostClassifier 의 정답률:  0.3152
# BaggingClassifier 의 정답률:  0.929
# BernoulliNB 의 정답률:  0.8615
# CalibratedClassifierCV 의 정답률:  0.9624
# CategoricalNB 은 안나온 놈
# ClassifierChain 은 안나온 놈
# ComplementNB 의 정답률:  0.8246
# DecisionTreeClassifier 의 정답률:  0.8184
# DummyClassifier 의 정답률:  0.0891
# ExtraTreeClassifier 의 정답률:  0.7725
# ExtraTreesClassifier 의 정답률:  0.9805
# GaussianNB 의 정답률:  0.8594
# GaussianProcessClassifier 의 정답률:  0.9784
# GradientBoostingClassifier 의 정답률:  0.9575
# HistGradientBoostingClassifier 의 정답률:  0.9736
# KNeighborsClassifier 의 정답률:  0.9826
# LabelPropagation 의 정답률:  0.9847
# LabelSpreading 의 정답률:  0.9847
# LinearDiscriminantAnalysis 의 정답률:  0.9506
# LinearSVC 의 정답률:  0.9624
# LogisticRegression 의 정답률:  0.961
# LogisticRegressionCV 의 정답률:  0.9666
# MLPClassifier 의 정답률:  0.9742
# MultiOutputClassifier 은 안나온 놈
# MultinomialNB 의 정답률:  0.8998
# NearestCentroid 의 정답률:  0.9033
# NuSVC 의 정답률:  0.9631
# OneVsOneClassifier 은 안나온 놈
# OneVsRestClassifier 은 안나온 놈
# OutputCodeClassifier 은 안나온 놈
# PassiveAggressiveClassifier 의 정답률:  0.952
# Perceptron 의 정답률:  0.9374
# QuadraticDiscriminantAnalysis 의 정답률:  0.8935
# RadiusNeighborsClassifier 은 안나온 놈
# RandomForestClassifier 의 정답률:  0.9708
# RidgeClassifier 의 정답률:  0.9311
# RidgeClassifierCV 의 정답률:  0.9311
# SGDClassifier 의 정답률:  0.9506
# SVC 의 정답률:  0.984
# StackingClassifier 은 안나온 놈
# VotingClassifier 은 안나온 놈