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
x_test = scl.fit_transform(x_test)



parameters_rnf = {'n_estimators':[400],'max_depth':[None],'min_samples_leaf':[1],'min_samples_split':[2],
}

# 2. 모델
xg = XGBRegressor(n_estimators=100, learning_rate=1, max_depth=2, gamma=0, min_child_weight=1, subsample=1, colsample_bytree=0.5, 
                   colsample_bylevel=1, colsample_bynode=1, reg_alpha=0.01,
                   tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=0, random_state=1234,)
lg = LGBMRegressor()
cat = CatBoostRegressor(verbose=0)
rf = RandomForestRegressor(n_estimators=400, max_depth=None, min_samples_leaf=1 ,min_samples_split=2)

model = VotingRegressor(estimators=[('XG', xg), ('LG', lg), ('CAT', cat), ('RF', rf)],
                        #  voting='hard'
                         )

# 3. 훈련
model.fit(x_train,y_train)

# 4. 평가, 예측
y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print('voting result: ', round(r2, 4))



classifiers = [xg, lg, cat, rf]
for model2 in classifiers:
    model2.fit(x_train, y_train)
    y_pred = model2.predict(x_test)
    r22 = r2_score(y_test, y_pred)
    class_name = model2.__class__.__name__
    print('{0} 정확도: {1:.4f}'.format(class_name, r22))
    
    
# voting result:  0.751
# XGBRegressor 정확도: 0.6063
# LGBMRegressor 정확도: 0.7374
# CatBoostRegressor 정확도: 0.7599
# RandomForestRegressor 정확도: 0.7397