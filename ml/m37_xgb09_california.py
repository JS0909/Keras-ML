import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_wine, load_digits
from sklearn.datasets import load_breast_cancer, fetch_covtype, fetch_california_housing
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from xgboost import XGBClassifier, XGBRegressor


# 1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
print(x.shape)
le = LabelEncoder()
y = le.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=1234, train_size=0.8)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1234)

parameters = {
            'n_estimators':[100],
            'learning_rate':[1],
            'max_depth':[None,2,3,4,5,6,7,8,9,10],
            'gamma':[0],
            'min_child_weight':[1],
            'subsample':[1],
            'colsample_bytree':[0,0.1,0.2,0.3,0.5,0.7,1] ,
            'colsample_bylevel':[1],
            'colsample_bynode':[0,0.1,0.2,0.3,0.5,0.7,1],
            'alpha':[0,0.1,0.01,0.001,1,2,10],
            'lambda':[0,0.1,0.01,0.001,1,2,10]
              }  

# 2. 모델
xgb = XGBRegressor(tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=0, random_state=1234)
model = RandomizedSearchCV(xgb, parameters, cv=kfold, n_jobs=-1, verbose=2)

model.fit(x_train, y_train)

print('최상의 매개변수: ', model.best_params_)
print('최상의 점수: ', model.best_score_)
print('테스트 스코어: ', model.score(x_test, y_test))

# 최상의 매개변수:  {'subsample': 1, 'n_estimators': 100, 'min_child_weight': 1, 'max_depth': 4, 'learning_rate': 1, 'lambda': 0.001, 'gamma': 0, 'colsample_bytree': 0.3, 'colsample_bynode': 1, 'colsample_bylevel': 1, 'alpha': 0.01}
# 최상의 점수:  0.7606849665250127
# 테스트 스코어:  0.7582096982868436

