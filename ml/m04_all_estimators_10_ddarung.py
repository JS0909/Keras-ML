# Dacon 따릉이 문제풀이
import pandas as pd
from pandas import DataFrame 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.python.keras.callbacks import EarlyStopping
import time
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
import numpy as np
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore') # warnig 출력 안함


# 1. 데이터
path = 'D:\study_data\_data\ddarung/'
train_set = pd.read_csv(path+'train.csv',index_col=0) # index_col = n : n번째 칼럼을 인덱스로 인식
# print(train_set)
# print(train_set.shape) # (1459, 10)

test_set = pd.read_csv(path+'test.csv', index_col=0)
# print(test_set)
# print(test_set.shape) # (715, 9)

### 결측치 처리(일단 제거로 처리) ###
print(train_set.info())
print(train_set.isnull().sum()) # 결측치 전부 더함
# train_set = train_set.dropna() # nan 값(결측치) 열 없앰
train_set = train_set.fillna(0) # 결측치 0으로 채움
print(train_set.isnull().sum()) # 없어졌는지 재확인

x = train_set.drop(['count'], axis=1) # axis = 0은 열방향으로 쭉 한줄(가로로 쭉), 1은 행방향으로 쭉 한줄(세로로 쭉)
y = train_set['count']

print(x.shape, y.shape) # (1328, 9) (1328,)

# trainset과 testset의 분리
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler() 
scaler = RobustScaler()
scaler.fit(x_train)
scaler.fit(test_set)
test_set = scaler.transform(test_set)
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

# 5. 제출 준비
# submission = pd.read_csv(path + 'submission.csv', index_col=0)
# y_submit = model.predict(test_set)
# submission['count'] = y_submit
# submission.to_csv(path + 'submission.csv', index=True)

# ARDRegression 의 정답률:  0.6105770566025065
# AdaBoostRegressor 의 정답률:  0.6503038553116387
# BaggingRegressor 의 정답률:  0.7536648581927926
# BayesianRidge 의 정답률:  0.6118640145366676
# CCA 의 정답률:  0.43315483473113847
# DecisionTreeRegressor 의 정답률:  0.6638154242240161
# DummyRegressor 의 정답률:  -0.0017219589708741267
# ElasticNet 의 정답률:  0.5205593122681362
# ElasticNetCV 의 정답률:  0.6049181419472062
# ExtraTreeRegressor 의 정답률:  0.6205354358102326
# ExtraTreesRegressor 의 정답률:  0.810687549856089
# GammaRegressor 의 정답률:  0.4719064411212013
# GaussianProcessRegressor 의 정답률:  0.46308360777960045
# GradientBoostingRegressor 의 정답률:  0.7841492578937992
# HistGradientBoostingRegressor 의 정답률:  0.7927032212210345
# HuberRegressor 의 정답률:  0.5937258588440054
# IsotonicRegression 은 안나온 놈
# KNeighborsRegressor 의 정답률:  0.6512741305620451
# KernelRidge 의 정답률:  -0.21855220556249733
# Lars 의 정답률:  0.6120638972205654
# LarsCV 의 정답률:  0.6097301220998342
# Lasso 의 정답률:  0.6039761280650496
# LassoCV 의 정답률:  0.6118696236001524
# LassoLars 의 정답률:  0.27388572941919775
# LassoLarsCV 의 정답률:  0.6097301220998342
# LassoLarsIC 의 정답률:  0.6112348243076315
# LinearRegression 의 정답률:  0.6120638972205652
# LinearSVR 의 정답률:  0.5124828832868416
# MLPRegressor 의 정답률:  0.5724252159411403
# MultiOutputRegressor 은 안나온 놈
# MultiTaskElasticNet 은 안나온 놈
# MultiTaskElasticNetCV 은 안나온 놈
# MultiTaskLasso 은 안나온 놈
# MultiTaskLassoCV 은 안나온 놈
# NuSVR 의 정답률:  0.40826502122432484
# OrthogonalMatchingPursuit 의 정답률:  0.37892445023971255
# OrthogonalMatchingPursuitCV 의 정답률:  0.5952312098199338
# PLSCanonical 의 정답률:  0.009237160674834155
# PLSRegression 의 정답률:  0.6068936616841095
# PassiveAggressiveRegressor 의 정답률:  0.5389200888681802
# PoissonRegressor 의 정답률:  0.6784062871793362
# RANSACRegressor 의 정답률:  0.503836222756734
# RadiusNeighborsRegressor 은 안나온 놈
# RandomForestRegressor 의 정답률:  0.7795845758378009
# RegressorChain 은 안나온 놈
# Ridge 의 정답률:  0.6120231132770069
# RidgeCV 의 정답률:  0.6120231132770886
# SGDRegressor 의 정답률:  0.611363655952911
# SVR 의 정답률:  0.4199089898113908
# StackingRegressor 은 안나온 놈
# TheilSenRegressor 의 정답률:  0.59599601808237
# TransformedTargetRegressor 의 정답률:  0.6120638972205652
# TweedieRegressor 의 정답률:  0.45609280981404765
# VotingRegressor 은 안나온 놈