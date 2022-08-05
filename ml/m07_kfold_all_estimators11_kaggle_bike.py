# Kaggle Bike_sharing
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.utils import all_estimators
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
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
        
# # 5. 제출 준비
# submission = pd.read_csv(path + 'submission.csv', index_col=0)
# y_submit = model.predict(test_set)
# submission['count'] = np.abs(y_submit) # 마이너스 나오는거 절대값 처리

# submission.to_csv(path + 'submission.csv', index=True)

# ARDRegression 의 정답률:  0.3821
# AdaBoostRegressor 의 정답률:  0.6617
# BaggingRegressor 의 정답률:  0.9369
# BayesianRidge 의 정답률:  0.382
# CCA 의 정답률:  0.1137
# DecisionTreeRegressor 의 정답률:  0.8898
# DummyRegressor 의 정답률:  -0.0007
# ElasticNet 의 정답률:  0.1347
# ElasticNetCV 의 정답률:  0.3679
# ExtraTreeRegressor 의 정답률:  0.8591
# ExtraTreesRegressor 의 정답률:  0.9471
# GammaRegressor 의 정답률:  0.0555
# GaussianProcessRegressor 의 정답률:  -40.188
# GradientBoostingRegressor 의 정답률:  0.8542
# HistGradientBoostingRegressor 의 정답률:  0.9513
# HuberRegressor 의 정답률:  0.35
# IsotonicRegression 은 안나온 놈
# KNeighborsRegressor 의 정답률:  0.6615
# KernelRidge 의 정답률:  0.3821
# Lars 의 정답률:  0.3558
# LarsCV 의 정답률:  0.3808
# Lasso 의 정답률:  0.3799
# LassoCV 의 정답률:  0.3817
# LassoLars 의 정답률:  -0.0007
# LassoLarsCV 의 정답률:  0.3816
# LassoLarsIC 의 정답률:  0.3819
# LinearRegression 의 정답률:  0.3818
# LinearSVR 의 정답률:  0.3064
# MLPRegressor 의 정답률:  0.4691
# MultiOutputRegressor 은 안나온 놈
# MultiTaskElasticNet 은 안나온 놈
# MultiTaskElasticNetCV 은 안나온 놈
# MultiTaskLasso 은 안나온 놈
# MultiTaskLassoCV 은 안나온 놈
# NuSVR 의 정답률:  0.3231
# OrthogonalMatchingPursuit 의 정답률:  0.1535
# OrthogonalMatchingPursuitCV 의 정답률:  0.3805
# PLSCanonical 의 정답률:  -0.3294
# PLSRegression 의 정답률:  0.3765
# PassiveAggressiveRegressor 의 정답률:  0.3287
# PoissonRegressor 의 정답률:  0.4094
# RANSACRegressor 의 정답률:  0.116
# RadiusNeighborsRegressor 의 정답률:  0.2149
# RandomForestRegressor 의 정답률:  0.9448
# RegressorChain 은 안나온 놈
# Ridge 의 정답률:  0.382
# RidgeCV 의 정답률:  0.382
# SGDRegressor 의 정답률:  0.3815
# SVR 의 정답률:  0.3141
# StackingRegressor 은 안나온 놈
# TheilSenRegressor 의 정답률:  0.3784
# TransformedTargetRegressor 의 정답률:  0.3818
# TweedieRegressor 의 정답률:  0.083
# VotingRegressor 은 안나온 놈