from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
import numpy as np
import pandas as pd
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore') # warnig 출력 안함

# 1. 데이터
datasets = load_diabetes()
print(datasets.DESCR)
print(datasets.feature_names)
x = datasets['data']
y = datasets['target']

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
        
# ARDRegression 의 정답률:  0.4317
# AdaBoostRegressor 의 정답률:  0.3419
# BaggingRegressor 의 정답률:  0.2687
# BayesianRidge 의 정답률:  0.431
# CCA 의 정답률:  0.4411
# DecisionTreeRegressor 의 정답률:  -0.3152
# DummyRegressor 의 정답률:  -0.0366
# ElasticNet 의 정답률:  0.0877
# ElasticNetCV 의 정답률:  0.4314
# ExtraTreeRegressor 의 정답률:  -0.1919
# ExtraTreesRegressor 의 정답률:  0.3503
# GammaRegressor 의 정답률:  0.0369
# GaussianProcessRegressor 의 정답률:  -8.4939
# GradientBoostingRegressor 의 정답률:  0.3095
# HistGradientBoostingRegressor 의 정답률:  0.2625
# HuberRegressor 의 정답률:  0.4322
# IsotonicRegression 은 안나온 놈
# KNeighborsRegressor 의 정답률:  0.3522
# KernelRidge 의 정답률:  0.4274
# Lars 의 정답률:  0.4302
# LarsCV 의 정답률:  0.4193
# Lasso 의 정답률:  0.4286
# LassoCV 의 정답률:  0.4166
# LassoLars 의 정답률:  0.3338
# LassoLarsCV 의 정답률:  0.4165
# LassoLarsIC 의 정답률:  0.4308
# LinearRegression 의 정답률:  0.4302
# LinearSVR 의 정답률:  0.1345
# MLPRegressor 의 정답률:  -0.7255
# MultiOutputRegressor 은 안나온 놈
# MultiTaskElasticNet 은 안나온 놈
# MultiTaskElasticNetCV 은 안나온 놈
# MultiTaskLasso 은 안나온 놈
# MultiTaskLassoCV 은 안나온 놈
# NuSVR 의 정답률:  0.0778
# OrthogonalMatchingPursuit 의 정답률:  0.2676
# OrthogonalMatchingPursuitCV 의 정답률:  0.4223
# PLSCanonical 의 정답률:  -1.3462
# PLSRegression 의 정답률:  0.4284
# PassiveAggressiveRegressor 의 정답률:  0.4257
# PoissonRegressor 의 정답률:  0.4263
# RANSACRegressor 의 정답률:  -0.0243
# RadiusNeighborsRegressor 의 정답률:  0.1069
# RandomForestRegressor 의 정답률:  0.3198
# RegressorChain 은 안나온 놈
# Ridge 의 정답률:  0.4328
# RidgeCV 의 정답률:  0.425
# SGDRegressor 의 정답률:  0.4276
# SVR 의 정답률:  0.0882
# StackingRegressor 은 안나온 놈
# TheilSenRegressor 의 정답률:  0.4224
# TransformedTargetRegressor 의 정답률:  0.4302
# TweedieRegressor 의 정답률:  0.0403
# VotingRegressor 은 안나온 놈