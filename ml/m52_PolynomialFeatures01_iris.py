from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, MinMaxScaler
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline

# 1. 데이터
datasets = load_iris()
x,y = datasets.data, datasets.target
print(x.shape, y.shape) # (150, 4) (150,)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)

# 2. 모델
model = make_pipeline(MinMaxScaler(), RandomForestClassifier())
model.fit(x_train, y_train)
print('그냥 스코어: ',model.score(x_test, y_test))

from sklearn.model_selection import cross_val_score
kfold = KFold(n_splits=5, shuffle=True, random_state=1234)
score = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')
print('CV: ', score)
print('CV n빵: ', np.mean(score))
print('-------------------------------------------------------------')
#============================ PolynomialFeatures 후 ================================
pf = PolynomialFeatures(degree=2, include_bias=False) # include_bias = False 하면 기본으로 생기는 1이 안나옴
xp = pf.fit_transform(x)
print(xp.shape) # (150, 14)

x_train, x_test, y_train, y_test = train_test_split(xp, y, test_size=0.2, random_state=1234)

# 2. 모델
model = make_pipeline(MinMaxScaler(), RandomForestClassifier())
model.fit(x_train, y_train)
print('폴리스코어: ', model.score(x_test, y_test))

score = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')
print('폴리 CV: ', score)
print('폴리 CV n빵: ', np.mean(score))


# 그냥 스코어:  1.0
# CV:  [0.91666667 0.83333333 1.         0.95833333 0.95833333]
# CV n빵:  0.9333333333333333
# -------------------------------------------------------------
# 폴리스코어:  1.0
# 폴리 CV:  [0.91666667 0.83333333 1.         0.95833333 0.95833333]
# 폴리 CV n빵:  0.9333333333333333