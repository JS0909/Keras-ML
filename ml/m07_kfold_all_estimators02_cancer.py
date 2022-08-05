from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore') # warnig 출력 안함

# 1. 데이터
datasets = load_breast_cancer()
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

# AdaBoostClassifier 의 정답률:  0.9648
# BaggingClassifier 의 정답률:  0.9385
# BernoulliNB 의 정답률:  0.611
# CalibratedClassifierCV 의 정답률:  0.9758
# CategoricalNB 은 안나온 놈
# ClassifierChain 은 안나온 놈
# ComplementNB 은 안나온 놈
# DecisionTreeClassifier 의 정답률:  0.9385
# DummyClassifier 의 정답률:  0.622
# ExtraTreeClassifier 의 정답률:  0.9077
# ExtraTreesClassifier 의 정답률:  0.9648
# GaussianNB 의 정답률:  0.9319
# GaussianProcessClassifier 의 정답률:  0.9582
# GradientBoostingClassifier 의 정답률:  0.9604
# HistGradientBoostingClassifier 의 정답률:  0.9604
# KNeighborsClassifier 의 정답률:  0.9692
# LabelPropagation 의 정답률:  0.9736
# LabelSpreading 의 정답률:  0.9714
# LinearDiscriminantAnalysis 의 정답률:  0.956
# LinearSVC 의 정답률:  0.9758
# LogisticRegression 의 정답률:  0.967
# LogisticRegressionCV 의 정답률:  0.9758
# MLPClassifier 의 정답률:  0.9736
# MultiOutputClassifier 은 안나온 놈
# MultinomialNB 은 안나온 놈
# NearestCentroid 의 정답률:  0.9407
# NuSVC 의 정답률:  0.9473
# OneVsOneClassifier 은 안나온 놈
# OneVsRestClassifier 은 안나온 놈
# OutputCodeClassifier 은 안나온 놈
# PassiveAggressiveClassifier 의 정답률:  0.9604
# Perceptron 의 정답률:  0.9604
# QuadraticDiscriminantAnalysis 의 정답률:  0.956
# RadiusNeighborsClassifier 의 정답률:  nan
# RandomForestClassifier 의 정답률:  0.9604
# RidgeClassifier 의 정답률:  0.9604
# RidgeClassifierCV 의 정답률:  0.9604
# SGDClassifier 의 정답률:  0.9692
# SVC 의 정답률:  0.9802
# StackingClassifier 은 안나온 놈
# VotingClassifier 은 안나온 놈