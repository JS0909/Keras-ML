from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, KFold,\
    HalvingRandomSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PolynomialFeatures
import warnings
warnings.filterwarnings('ignore') # warnig 출력 안함

import numpy as np
import pandas as pd

from sklearn.ensemble import VotingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
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

print(x.shape, y.shape) # (1459, 9) (1459,)
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
print(xp.shape) # (1459, 54)

x_train, x_test, y_train, y_test = train_test_split(xp, y, test_size=0.2, random_state=13)

# 2. 모델
model = make_pipeline(MinMaxScaler(), LinearRegression())
model.fit(x_train, y_train)
print('폴리스코어: ', model.score(x_test, y_test))

score = cross_val_score(model, x_train, y_train, cv=kfold, scoring='r2')
print('폴리 CV: ', score)
print('폴리 CV n빵: ', np.mean(score))

# 그냥 스코어:  0.577068971279496
# CV:  [0.65290683 0.44922132 0.60175833 0.63353471 0.59770609]
# CV n빵:  0.5870254564136671
# -------------------------------------------------------------
# (1459, 54)
# 폴리스코어:  0.6377358814798159
# 폴리 CV:  [0.63531897 0.4845633  0.6674983  0.65102733 0.59192098]
# 폴리 CV n빵:  0.6060657766753408