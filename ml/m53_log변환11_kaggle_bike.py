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

from sklearn.ensemble import VotingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

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

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)

# scl = StandardScaler()
# x_train = scl.fit_transform(x_train)
# x_test = scl.transform(x_test)

# 2. 모델
# model = LinearRegression()
model = RandomForestRegressor(random_state=1234)

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
y_pred = model.predict(x_test)
result = r2_score(y_test, y_pred)
print('그냥 결과: ', round(result,4))

# 그냥 결과:  0.9538

#=================== 로그 변환 ======================
print(x.columns)

import matplotlib.pyplot as plt
x.plot.box()
plt.title('bike')
plt.xlabel('Feature')
plt.ylabel('데이터값')
# plt.show()

# x['windspeed'] = np.log1p(x['windspeed']) # 0.9537
x['holiday'] = np.log1p(x['holiday']) # 0.9538
# 둘다 0.9537

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)

# scl = StandardScaler()
# x_train = scl.fit_transform(x_train)
# x_test = scl.transform(x_test)

# 2. 모델
# model = LinearRegression()
model = RandomForestRegressor(random_state=1234)

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
y_pred = model.predict(x_test)
result = r2_score(y_test, y_pred)
print('로그 변환 결과: ', round(result,4))