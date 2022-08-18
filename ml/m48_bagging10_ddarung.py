from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, KFold,\
    HalvingRandomSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import warnings
warnings.filterwarnings('ignore') # warnig 출력 안함

import numpy as np
import pandas as pd

from sklearn.ensemble import VotingRegressor, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# 1. 데이터
path = 'D:\study_data\_data\ddarung/'
train_set = pd.read_csv(path+'train.csv',index_col=0) # index_col = n : n번째 칼럼을 인덱스로 인식
# print(train_set)
# print(train_set.shape) # (1459, 10)

test_set = pd.read_csv(path+'test.csv', index_col=0)
# print(test_set)
# print(test_set.shape) # (715, 9)

### 결측치 중간값으로 ###
print(train_set.info())
print(train_set.isnull().sum()) # 결측치 전부 더함
median = train_set.median()
train_set = train_set.fillna(median) # 결측치 중간값으로
print(train_set.isnull().sum()) # 없어졌는지 재확인

x = train_set.drop(['count'], axis=1) # axis = 0은 열방향으로 쭉 한줄(가로로 쭉), 1은 행방향으로 쭉 한줄(세로로 쭉)
y = train_set['count']

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, random_state=704)

scl = MinMaxScaler()
x_train = scl.fit_transform(x_train)
x_test = scl.transform(x_test)


# 2. 모델
from sklearn.ensemble import BaggingRegressor
from xgboost import XGBRegressor

xgb = XGBRegressor(tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=0, random_state=1234)
model = BaggingRegressor(xgb, n_estimators=100, n_jobs=-1, random_state=1234)

# 3. 훈련
model.fit(x_train, y_train)


# 4. 평가, 예측
print(model.score(x_test, y_test))

# 0.7847884681388737