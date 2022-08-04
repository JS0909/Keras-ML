from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
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
             
#2. 모델구성
# allAlgorithms = all_estimators(type_filter='classifier')
allAlgorithms = all_estimators(type_filter='regressor')
print('allAlgorithms: ', allAlgorithms)
print('모델의 개수: ', len(allAlgorithms)) # 54

for (name, algorithm) in allAlgorithms:
    try:
        model = algorithm()
        model.fit(x_train, y_train)
        ypred = model.predict(x_test)
        r2 = r2_score(y_test, ypred)
        print(name, '의 정답률: ', r2)
    except:
        # continue # 또는 pass
        print(name, '은 안나온 놈')
        
# ARDRegression 의 정답률:  0.5894071989795238
# AdaBoostRegressor 의 정답률:  0.5121603517958673
# BaggingRegressor 의 정답률:  0.47335355996195605
# BayesianRidge 의 정답률:  0.5954128350161362
# CCA 의 정답률:  0.5852713803269576
# DecisionTreeRegressor 의 정답률:  -0.2139904943831099
# DummyRegressor 의 정답률:  -0.01545589029660177
# ElasticNet 의 정답률:  0.1436197162111983
# ElasticNetCV 의 정답률:  0.5838811172714529
# ExtraTreeRegressor 의 정답률:  -0.0892319985472052
# ExtraTreesRegressor 의 정답률:  0.5435063756542552
# GammaRegressor 의 정답률:  0.08710574754327771
# GaussianProcessRegressor 의 정답률:  -16.860774092820694
# GradientBoostingRegressor 의 정답률:  0.5467392508907825
# HistGradientBoostingRegressor 의 정답률:  0.5392377257708727
# HuberRegressor 의 정답률:  0.5785993482753421
# IsotonicRegression 은 안나온 놈
# KNeighborsRegressor 의 정답률:  0.4904172186988308
# KernelRidge 의 정답률:  0.5905597611712815
# Lars 의 정답률:  0.585114126995974
# LarsCV 의 정답률:  0.5896206660679899
# Lasso 의 정답률:  0.5816741737120826
# LassoCV 의 정답률:  0.5864701793206931
# LassoLars 의 정답률:  0.4523872639391451
# LassoLarsCV 의 정답률:  0.5896206660679899
# LassoLarsIC 의 정답률:  0.5962837270477235
# LinearRegression 의 정답률:  0.5851141269959738
# LinearSVR 의 정답률:  0.351068281251975
# MLPRegressor 의 정답률:  -0.284580935709015
# MultiOutputRegressor 은 안나온 놈
# MultiTaskElasticNet 은 안나온 놈
# MultiTaskElasticNetCV 은 안나온 놈
# MultiTaskLasso 은 안나온 놈
# MultiTaskLassoCV 은 안나온 놈
# NuSVR 의 정답률:  0.16780359645796983
# OrthogonalMatchingPursuit 의 정답률:  0.32241768669099424
# OrthogonalMatchingPursuitCV 의 정답률:  0.5812114430406671
# PLSCanonical 의 정답률:  -1.6878518911601064
# PLSRegression 의 정답률:  0.6072433305368464
# PassiveAggressiveRegressor 의 정답률:  0.5617070726539444
# PoissonRegressor 의 정답률:  0.5800995236270168
# RANSACRegressor 의 정답률:  0.16265851506420703
# RadiusNeighborsRegressor 의 정답률:  0.17842231317289414
# RandomForestRegressor 의 정답률:  0.5592396647121152
# RegressorChain 은 안나온 놈
# Ridge 의 정답률:  0.5940247828297052
# RidgeCV 의 정답률:  0.5935251107764845
# SGDRegressor 의 정답률:  0.5818649045655409
# SVR 의 정답률:  0.18982394795986235
# StackingRegressor 은 안나온 놈
# TheilSenRegressor 의 정답률:  0.5848407868028076
# TransformedTargetRegressor 의 정답률:  0.5851141269959738
# TweedieRegressor 의 정답률:  0.0830837969884558
# VotingRegressor 은 안나온 놈