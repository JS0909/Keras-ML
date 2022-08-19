from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, MinMaxScaler
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline

# 1. 데이터
datasets = fetch_california_housing()
x,y = datasets.data, datasets.target
print(x.shape, y.shape) # (20640, 8) (20640,)

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
print(xp.shape) # (20640, 44)

x_train, x_test, y_train, y_test = train_test_split(xp, y, test_size=0.2, random_state=13)

# 2. 모델
model = make_pipeline(MinMaxScaler(), LinearRegression())
model.fit(x_train, y_train)
print('폴리스코어: ', model.score(x_test, y_test))

score = cross_val_score(model, x_train, y_train, cv=kfold, scoring='r2')
print('폴리 CV: ', score)
print('폴리 CV n빵: ', np.mean(score))


# 그냥 스코어:  0.6047072598918702
# CV:  [0.62122413 0.5990505  0.57542312 0.62932938 0.59690546]
# CV n빵:  0.6043865160513695
# -------------------------------------------------------------
# 폴리스코어:  0.6517614771513793
# 폴리 CV:  [  0.68035964   0.65577963 -14.21463811   0.70059565   0.67612095]
# 폴리 CV n빵:  -2.300356451152336