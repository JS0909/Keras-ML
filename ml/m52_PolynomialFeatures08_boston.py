from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# 1. 데이터
datasets = load_boston()
x,y = datasets.data, datasets.target
print(x.shape, y.shape) # (506, 13) (506,)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)

# 2. 모델
model = make_pipeline(StandardScaler(), LinearRegression())
model.fit(x_train, y_train)
print('그냥 스코어: ',model.score(x_test, y_test))

from sklearn.model_selection import cross_val_score
kfold = KFold(n_splits=5, shuffle=True, random_state=1234)
score = cross_val_score(model, x_train, y_train, cv=kfold, scoring='r2')
print('CV: ', score)
print('CV n빵: ', np.mean(score))

# 그냥 스코어:  0.7665382927362877
# CV:  [0.71606004 0.67832011 0.65400513 0.56791147 0.7335664 ]
# CV n빵:  0.669972627809433

#============================ PolynomialFeatures 후 ================================
pf = PolynomialFeatures(degree=2, include_bias=False) # include_bias = False 하면 기본으로 생기는 1이 안나옴
xp = pf.fit_transform(x)
print(xp.shape) # (506, 105)

x_train, x_test, y_train, y_test = train_test_split(xp, y, test_size=0.2, random_state=1234)

# 2. 모델
model = make_pipeline(StandardScaler(), LinearRegression())
model.fit(x_train, y_train)
print('폴리스코어: ', model.score(x_test, y_test))

score = cross_val_score(model, x_train, y_train, cv=kfold, scoring='r2')
print('폴리 CV: ', score)
print('폴리 CV n빵: ', np.mean(score))

# 폴리스코어:  0.8745129304823852
# 폴리 CV:  [0.7917776  0.8215846  0.79599441 0.81776798 0.81170102]
# 폴리 CV n빵:  0.807765121221582