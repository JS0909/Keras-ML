import pandas as pd 
import numpy as np
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, KFold,\
    HalvingRandomSearchCV
from sklearn.metrics import r2_score
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PolynomialFeatures
import warnings
warnings.filterwarnings('ignore') # warnig 출력 안함

from sklearn.metrics import r2_score
from sklearn.ensemble import VotingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

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

print(x.shape, y.shape) # (10886, 12) (10886,)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=13)

# 2. 모델
model = make_pipeline(MinMaxScaler(), LinearRegression())
model.fit(x_train, y_train)
print('그냥 스코어: ',model.score(x_test, y_test))

from sklearn.model_selection import cross_val_score
kfold = KFold(n_splits=5, shuffle=True, random_state=1234)
score = cross_val_score(model, x_train, y_train, cv=kfold, scoring='r2')
print('CV: ', score)
print('CV n빵: ', np.mean(score))
print('-------------------------------------------------------------')
#============================ PolynomialFeatures 후 ================================
pf = PolynomialFeatures(degree=2, include_bias=False) # include_bias = False 하면 기본으로 생기는 1이 안나옴
xp = pf.fit_transform(x)
print(xp.shape) # (10886, 90)

x_train, x_test, y_train, y_test = train_test_split(xp, y, test_size=0.2, random_state=13)

# 2. 모델
model = make_pipeline(MinMaxScaler(), LinearRegression())
model.fit(x_train, y_train)
print('폴리스코어: ', model.score(x_test, y_test))

score = cross_val_score(model, x_train, y_train, cv=kfold, scoring='r2')
print('폴리 CV: ', score)
print('폴리 CV n빵: ', np.mean(score))

# 그냥 스코어:  0.3671828309253957
# CV:  [0.38359012 0.37200022 0.36645715 0.41022736 0.4226948 ]
# CV n빵:  0.3909939278874492
# -------------------------------------------------------------
# 폴리스코어:  0.5357049845955479
# 폴리 CV:  [0.54016536 0.54824101 0.51928361 0.55109354 0.57646241]
# 폴리 CV n빵:  0.5470491872231942