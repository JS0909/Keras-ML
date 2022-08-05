# Dacon 따릉이 문제풀이
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
from sklearn.utils import all_estimators
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
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

# 5. 제출 준비
# submission = pd.read_csv(path + 'submission.csv', index_col=0)
# y_submit = model.predict(test_set)
# submission['count'] = y_submit
# submission.to_csv(path + 'submission.csv', index=True)

# ARDRegression 의 정답률:  0.6054
# AdaBoostRegressor 의 정답률:  0.6028
# BaggingRegressor 의 정답률:  0.7642
# BayesianRidge 의 정답률:  0.6058
# CCA 의 정답률:  0.2289
# DecisionTreeRegressor 의 정답률:  0.5636
# DummyRegressor 의 정답률:  -0.0052
# ElasticNet 의 정답률:  0.2097
# ElasticNetCV 의 정답률:  0.5881
# ExtraTreeRegressor 의 정답률:  0.5293
# ExtraTreesRegressor 의 정답률:  0.7947
# GammaRegressor 의 정답률:  0.0946
# GaussianProcessRegressor 의 정답률:  -125.4578
# GradientBoostingRegressor 의 정답률:  0.7634
# HistGradientBoostingRegressor 의 정답률:  0.7838
# HuberRegressor 의 정답률:  0.5954
# IsotonicRegression 은 안나온 놈
# KNeighborsRegressor 의 정답률:  0.6993
# KernelRidge 의 정답률:  0.6066
# Lars 의 정답률:  0.5679
# LarsCV 의 정답률:  0.5803
# Lasso 의 정답률:  0.5836
# LassoCV 의 정답률:  0.6052
# LassoLars 의 정답률:  0.3681
# LassoLarsCV 의 정답률:  0.6055
# LassoLarsIC 의 정답률:  0.6058
# LinearRegression 의 정답률:  0.6056
# LinearSVR 의 정답률:  0.4705
# MLPRegressor 의 정답률:  0.4607
# MultiOutputRegressor 은 안나온 놈
# MultiTaskElasticNet 은 안나온 놈
# MultiTaskElasticNetCV 은 안나온 놈
# MultiTaskLasso 은 안나온 놈
# MultiTaskLassoCV 은 안나온 놈
# NuSVR 의 정답률:  0.4724
# OrthogonalMatchingPursuit 의 정답률:  0.3968
# OrthogonalMatchingPursuitCV 의 정답률:  0.5954
# PLSCanonical 의 정답률:  -0.3571
# PLSRegression 의 정답률:  0.6043
# PassiveAggressiveRegressor 의 정답률:  0.5833
# PoissonRegressor 의 정답률:  0.6185
# RANSACRegressor 의 정답률:  0.4708
# RadiusNeighborsRegressor 의 정답률:  0.3062
# RandomForestRegressor 의 정답률:  0.7819
# RegressorChain 은 안나온 놈
# Ridge 의 정답률:  0.6058
# RidgeCV 의 정답률:  0.6058
# SGDRegressor 의 정답률:  0.6046
# SVR 의 정답률:  0.466
# StackingRegressor 은 안나온 놈
# TheilSenRegressor 의 정답률:  0.5916
# TransformedTargetRegressor 의 정답률:  0.6056
# TweedieRegressor 의 정답률:  0.1287
# VotingRegressor 은 안나온 놈