from sklearn.datasets import fetch_california_housing
import numpy as np
import pandas as pd
from sklearn.utils import all_estimators
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
import warnings
warnings.filterwarnings('ignore') # warnig 출력 안함

# 1. 데이터
datasets = fetch_california_housing()
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
allAlgorithms = all_estimators(type_filter='regressor')
print('allAlgorithms: ', allAlgorithms)
print('모델의 개수: ', len(allAlgorithms))

for (name, algorithm) in allAlgorithms:
    try:
        model = algorithm()
        ypred = cross_val_predict(model, x_test, y_test, cv=kfold)
        score = cross_val_score(model, x_train, y_train, cv=kfold)
        print(name, '의 정답률: ', round(np.mean(score),4))
        
    except:
        # continue # 또는 pass
        print(name, '은 안나온 놈')
        
# ARDRegression 의 정답률:  0.6101
# AdaBoostRegressor 의 정답률:  0.4362
# BaggingRegressor 의 정답률:  0.7834
# BayesianRidge 의 정답률:  0.6099
# CCA 의 정답률:  0.5655
# DecisionTreeRegressor 의 정답률:  0.596
# DummyRegressor 의 정답률:  -0.0005
# ElasticNet 의 정답률:  -0.0005
# ElasticNetCV 의 정답률:  0.6012
# ExtraTreeRegressor 의 정답률:  0.5444
# ExtraTreesRegressor 의 정답률:  0.8115
# GammaRegressor 의 정답률:  0.0189
# GaussianProcessRegressor 의 정답률:  -14011.6662
# GradientBoostingRegressor 의 정답률:  0.787
# HistGradientBoostingRegressor 의 정답률:  0.8336
# HuberRegressor 의 정답률:  0.575
# IsotonicRegression 은 안나온 놈
# KNeighborsRegressor 의 정답률:  0.701
# KernelRidge 의 정답률:  0.5341
# Lars 의 정답률:  0.6099
# LarsCV 의 정답률:  0.6093
# Lasso 의 정답률:  -0.0005
# LassoCV 의 정답률:  0.6092
# LassoLars 의 정답률:  -0.0005
# LassoLarsCV 의 정답률:  0.6093
# LassoLarsIC 의 정답률:  0.6099
# LinearRegression 의 정답률:  0.6099
# LinearSVR 의 정답률:  0.585
# MLPRegressor 의 정답률:  0.7226
# MultiOutputRegressor 은 안나온 놈
# MultiTaskElasticNet 은 안나온 놈
# MultiTaskElasticNetCV 은 안나온 놈
# MultiTaskLasso 은 안나온 놈
# MultiTaskLassoCV 은 안나온 놈
# NuSVR 의 정답률:  0.6649
# OrthogonalMatchingPursuit 의 정답률:  0.4751
# OrthogonalMatchingPursuitCV 의 정답률:  0.6013
# PLSCanonical 의 정답률:  0.3688
# PLSRegression 의 정답률:  0.5235
# PassiveAggressiveRegressor 의 정답률:  -0.3942
# PoissonRegressor 의 정답률:  0.0409
# RANSACRegressor 의 정답률:  -0.6768
# RadiusNeighborsRegressor 의 정답률:  0.0139
# RandomForestRegressor 의 정답률:  0.8059
# RegressorChain 은 안나온 놈
# Ridge 의 정답률:  0.6032
# RidgeCV 의 정답률:  0.6076
# SGDRegressor 의 정답률:  0.5628
# SVR 의 정답률:  0.6612
# StackingRegressor 은 안나온 놈
# TheilSenRegressor 의 정답률:  -3.2995
# TransformedTargetRegressor 의 정답률:  0.6099
# TweedieRegressor 의 정답률:  0.019
# VotingRegressor 은 안나온 놈