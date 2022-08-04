from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
import numpy as np
import pandas as pd
from sklearn.utils import all_estimators
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
             
#2. 모델구성
# allAlgorithms = all_estimators(type_filter='classifier')
allAlgorithms = all_estimators(type_filter='regressor')
print('allAlgorithms: ', allAlgorithms)
print('모델의 개수: ', len(allAlgorithms)) # 41

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
        
# ARDRegression 의 정답률:  0.5745800077999317
# AdaBoostRegressor 의 정답률:  0.4080943421825277
# BaggingRegressor 의 정답률:  0.7958743492516143
# BayesianRidge 의 정답률:  0.5748327919383693
# CCA 의 정답률:  0.5405037912629643
# DecisionTreeRegressor 의 정답률:  0.6315428164771282
# DummyRegressor 의 정답률:  -0.00011902851648626367
# ElasticNet 의 정답률:  -0.00011902851648626367
# ElasticNetCV 의 정답률:  0.5912576265226432
# ExtraTreeRegressor 의 정답률:  0.531358886681981
# ExtraTreesRegressor 의 정답률:  0.8155218406796978
# GammaRegressor 의 정답률:  0.018695278742601795
# GaussianProcessRegressor 의 정답률:  -326790.4708505993
# GradientBoostingRegressor 의 정답률:  0.7839316527506819
# HistGradientBoostingRegressor 의 정답률:  0.8398968847975159
# HuberRegressor 의 정답률:  0.41737766126153
# IsotonicRegression 은 안나온 놈
# KNeighborsRegressor 의 정답률:  0.6983777673191758
# KernelRidge 의 정답률:  0.519858748869369
# Lars 의 정답률:  0.5743251711424028
# LarsCV 의 정답률:  0.5759929435416189
# Lasso 의 정답률:  -0.00011902851648626367
# LassoCV 의 정답률:  0.582567041263201
# LassoLars 의 정답률:  -0.00011902851648626367
# LassoLarsCV 의 정답률:  0.5759929435416189
# LassoLarsIC 의 정답률:  0.5744396460612611
# LinearRegression 의 정답률:  0.5743251711424028
# LinearSVR 의 정답률:  0.5677871040034257
# MLPRegressor 의 정답률:  0.7106505017912808
# MultiOutputRegressor 은 안나온 놈
# MultiTaskElasticNet 은 안나온 놈
# MultiTaskElasticNetCV 은 안나온 놈
# MultiTaskLasso 은 안나온 놈
# MultiTaskLassoCV 은 안나온 놈
# NuSVR 의 정답률:  0.6622695786979397
# OrthogonalMatchingPursuit 의 정답률:  0.46544181563032083
# OrthogonalMatchingPursuitCV 의 정답률:  0.5817412520003353
# PLSCanonical 의 정답률:  0.3501264261158309
# PLSRegression 의 정답률:  0.5172809086813477
# PassiveAggressiveRegressor 의 정답률:  -0.0508960404975185
# PoissonRegressor 의 정답률:  0.03825642055362177
# RANSACRegressor 의 정답률:  -9.793615218945428
# RadiusNeighborsRegressor 은 안나온 놈
# RandomForestRegressor 의 정답률:  0.8129139542918545
# RegressorChain 은 안나온 놈
# Ridge 의 정답률:  0.5909754999790872
# RidgeCV 의 정답률:  0.581481375752776
# SGDRegressor 의 정답률:  0.5597091622010025
# SVR 의 정답률:  0.6570541150865892
# StackingRegressor 은 안나온 놈
# TheilSenRegressor 의 정답률:  -33.716830831272745
# TransformedTargetRegressor 의 정답률:  0.5743251711424028
# TweedieRegressor 의 정답률:  0.01890699636752957
# VotingRegressor 은 안나온 놈