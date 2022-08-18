import pandas as pd 
import numpy as np
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, KFold,\
    HalvingRandomSearchCV
from sklearn.metrics import r2_score
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import warnings
warnings.filterwarnings('ignore') # warnig 출력 안함

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
x = np.array(x)

print(x.shape, y.shape) # (10886, 14) (10886,)

allfeature = round(x.shape[1]*0.2, 0)
print('자를 갯수: ', int(allfeature))


x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, train_size=0.8, random_state=1234)

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
    


# voting result:  0.9495
# XGBRegressor 정확도: 0.8238
# LGBMRegressor 정확도: 0.9549
# CatBoostRegressor 정확도: 0.9554
# RandomForestRegressor 정확도: 0.9532