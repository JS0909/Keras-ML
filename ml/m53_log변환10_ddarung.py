from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, KFold,\
    HalvingRandomSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PolynomialFeatures
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore') # warnig 출력 안함

import numpy as np
import pandas as pd

from sklearn.ensemble import VotingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
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

# 그냥 결과:  0.7887

#=================== 로그 변환 ======================
print(x.columns)

import matplotlib.pyplot as plt
x.plot.box()
plt.title('ddarung')
plt.xlabel('Feature')
plt.ylabel('데이터값')
# plt.show()

x['hour_bef_pm10'] = np.log1p(x['hour_bef_pm10']) # 0.7891
x['hour_bef_pm2.5'] = np.log1p(x['hour_bef_pm2.5']) # 0.7887
# 둘다 0.7892



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