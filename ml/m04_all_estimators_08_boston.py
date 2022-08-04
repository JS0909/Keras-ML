from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
import numpy as np
import pandas as pd
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore') # warnig 출력 안함

# 1. 데이터
datasets = load_boston()
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
        
# ARDRegression 의 정답률:  0.759510748393996
# AdaBoostRegressor 의 정답률:  0.854603126935245
# BaggingRegressor 의 정답률:  0.8623406514045568
# BayesianRidge 의 정답률:  0.7643584901441085
# CCA 의 정답률:  0.7439478352580443
# DecisionTreeRegressor 의 정답률:  0.8405320408899686
# DummyRegressor 의 정답률:  -9.191112707918059e-05
# ElasticNet 의 정답률:  0.14957299225612675
# ElasticNetCV 의 정답률:  0.7613046921246275
# ExtraTreeRegressor 의 정답률:  0.7851434049113846
# ExtraTreesRegressor 의 정답률:  0.9016072125557432
# GammaRegressor 의 정답률:  0.18804621694563972
# GaussianProcessRegressor 의 정답률:  -0.8553906219883416
# GradientBoostingRegressor 의 정답률:  0.895052205047173
# HistGradientBoostingRegressor 의 정답률:  0.8974347384568234
# HuberRegressor 의 정답률:  0.722607944920151
# IsotonicRegression 은 안나온 놈
# KNeighborsRegressor 의 정답률:  0.792220074153277
# KernelRidge 의 정답률:  0.6721155720002827
# Lars 의 정답률:  0.766365568566984
# LarsCV 의 정답률:  0.7597113125353329
# Lasso 의 정답률:  0.20861888462938227
# LassoCV 의 정답률:  0.7646785658807344
# LassoLars 의 정답률:  -9.191112707918059e-05
# LassoLarsCV 의 정답률:  0.7637160540796171
# LassoLarsIC 의 정답률:  0.7626793302700182
# LinearRegression 의 정답률:  0.7660111574904014
# LinearSVR 의 정답률:  0.645483449429678
# MLPRegressor 의 정답률:  0.3939840162055961
# MultiOutputRegressor 은 안나온 놈
# MultiTaskElasticNet 은 안나온 놈
# MultiTaskElasticNetCV 은 안나온 놈
# MultiTaskLasso 은 안나온 놈
# MultiTaskLassoCV 은 안나온 놈
# NuSVR 의 정답률:  0.5956889777053342
# OrthogonalMatchingPursuit 의 정답률:  0.5514649699541575
# OrthogonalMatchingPursuitCV 의 정답률:  0.7084085339582888
# PLSCanonical 의 정답률:  -1.5147647330085796
# PLSRegression 의 정답률:  0.7256007470235619
# PassiveAggressiveRegressor 의 정답률:  0.7258249066675032
# PoissonRegressor 의 정답률:  0.6598063866500516
# RANSACRegressor 의 정답률:  0.53632901642347
# RadiusNeighborsRegressor 의 정답률:  0.41598410356140314
# RandomForestRegressor 의 정답률:  0.8831973931871863
# RegressorChain 은 안나온 놈
# Ridge 의 정답률:  0.7580352036166451
# RidgeCV 의 정답률:  0.7653105244708671
# SGDRegressor 의 정답률:  0.7400571587373288
# SVR 의 정답률:  0.6159601090773135
# StackingRegressor 은 안나온 놈
# TheilSenRegressor 의 정답률:  0.7133554396734398
# TransformedTargetRegressor 의 정답률:  0.7660111574904014
# TweedieRegressor 의 정답률:  0.18609184468174544
# VotingRegressor 은 안나온 놈