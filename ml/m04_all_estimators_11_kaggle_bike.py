# Kaggle Bike_sharing
import numpy as np
import pandas as pd 
from pandas import DataFrame 
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.python.keras.callbacks import EarlyStopping
import time
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore') # warnig 출력 안함

# 1. 데이터
path = 'D:\study_data\_data\kaggle_bike/'
train_set = pd.read_csv(path+'train.csv')
# print(train_set)
# print(train_set.shape) # (10886, 11)

test_set = pd.read_csv(path+'test.csv')
# print(test_set)
# print(test_set.shape) # (6493, 8)

# datetime 열 내용을 각각 년월일시간날짜로 분리시켜 새 열들로 생성 후 원래 있던 datetime 열을 통째로 drop
train_set["hour"] = [t.hour for t in pd.DatetimeIndex(train_set.datetime)]
train_set["day"] = [t.dayofweek for t in pd.DatetimeIndex(train_set.datetime)]
train_set["month"] = [t.month for t in pd.DatetimeIndex(train_set.datetime)]
train_set['year'] = [t.year for t in pd.DatetimeIndex(train_set.datetime)]
train_set['year'] = train_set['year'].map({2011:0, 2012:1})

test_set["hour"] = [t.hour for t in pd.DatetimeIndex(test_set.datetime)]
test_set["day"] = [t.dayofweek for t in pd.DatetimeIndex(test_set.datetime)]
test_set["month"] = [t.month for t in pd.DatetimeIndex(test_set.datetime)]
test_set['year'] = [t.year for t in pd.DatetimeIndex(test_set.datetime)]
test_set['year'] = test_set['year'].map({2011:0, 2012:1})

train_set.drop('datetime',axis=1,inplace=True) # train_set에서 데이트타임 드랍
test_set.drop('datetime',axis=1,inplace=True) # test_set에서 데이트타임 드랍
train_set.drop('casual',axis=1,inplace=True) # casul 드랍 이유 모르겠음
train_set.drop('registered',axis=1,inplace=True) # registered 드랍 이유 모르겠음

#print(train_set.info())
# null값이 없으므로 결측치 삭제과정 생략

x = train_set.drop(['count'], axis=1)
y = train_set['count']

print(x.shape, y.shape) # (10886, 14) (10886,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)
scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler() 
# scaler = RobustScaler()
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
        
# # 5. 제출 준비
# submission = pd.read_csv(path + 'submission.csv', index_col=0)
# y_submit = model.predict(test_set)
# submission['count'] = np.abs(y_submit) # 마이너스 나오는거 절대값 처리

# submission.to_csv(path + 'submission.csv', index=True)

# ARDRegression 의 정답률:  0.40230858934458247
# AdaBoostRegressor 의 정답률:  0.6897912107059598
# BaggingRegressor 의 정답률:  0.9438393991557792
# BayesianRidge 의 정답률:  0.40195851601063115
# CCA 의 정답률:  0.17174194597379666
# DecisionTreeRegressor 의 정답률:  0.897705620169432
# DummyRegressor 의 정답률:  -0.00039203162920076196
# ElasticNet 의 정답률:  0.13721111857465518
# ElasticNetCV 의 정답률:  0.3841811406357608
# ExtraTreeRegressor 의 정답률:  0.8700928819254349
# ExtraTreesRegressor 의 정답률:  0.9502436480277134
# GammaRegressor 의 정답률:  0.08158319104414735
# GaussianProcessRegressor 의 정답률:  -26.065348412481665
# GradientBoostingRegressor 의 정답률:  0.8567197683226253
# HistGradientBoostingRegressor 의 정답률:  0.9567131684693747
# HuberRegressor 의 정답률:  0.365978805628717
# IsotonicRegression 은 안나온 놈
# KNeighborsRegressor 의 정답률:  0.6958239840616911
# KernelRidge 의 정답률:  0.40168854303978174
# Lars 의 정답률:  0.39962959815725607
# LarsCV 의 정답률:  0.4003005303210908
# Lasso 의 정답률:  0.39716956455997954
# LassoCV 의 정답률:  0.40225802708974767
# LassoLars 의 정답률:  -0.00039203162920076196
# LassoLarsCV 의 정답률:  0.40229161543997705
# LassoLarsIC 의 정답률:  0.40212404889799913
# LinearRegression 의 정답률:  0.40195970911693824
# LinearSVR 의 정답률:  0.325909934419783
# MLPRegressor 의 정답률:  0.5146616128429604
# MultiOutputRegressor 은 안나온 놈
# MultiTaskElasticNet 은 안나온 놈
# MultiTaskElasticNetCV 은 안나온 놈
# MultiTaskLasso 은 안나온 놈
# MultiTaskLassoCV 은 안나온 놈
# NuSVR 의 정답률:  0.3600365999559192
# OrthogonalMatchingPursuit 의 정답률:  0.17376893970338736
# OrthogonalMatchingPursuitCV 의 정답률:  0.40110337795664885
# PLSCanonical 의 정답률:  -0.23603465115462385
# PLSRegression 의 정답률:  0.39630119453059975
# PassiveAggressiveRegressor 의 정답률:  0.3228787669646648
# PoissonRegressor 의 정답률:  0.4120157352304761
# RANSACRegressor 의 정답률:  -0.050480243038036665
# RadiusNeighborsRegressor 의 정답률:  0.23274925060874152
# RandomForestRegressor 의 정답률:  0.9487075881635092
# RegressorChain 은 안나온 놈
# Ridge 의 정답률:  0.4019632343241908
# RidgeCV 의 정답률:  0.4019632343242716
# SGDRegressor 의 정답률:  0.40271891553851336
# SVR 의 정답률:  0.3485674007914782
# StackingRegressor 은 안나온 놈
# TheilSenRegressor 의 정답률:  0.398957072892126
# TransformedTargetRegressor 의 정답률:  0.40195970911693824
# TweedieRegressor 의 정답률:  0.08435056414597064
# VotingRegressor 은 안나온 놈