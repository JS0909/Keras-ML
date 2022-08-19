from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, MinMaxScaler
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline

# 1. 데이터
datasets = load_wine()
x,y = datasets.data, datasets.target
print(x.shape, y.shape) # (178, 13) (178,)

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
print(xp.shape) # (178, 104)

x_train, x_test, y_train, y_test = train_test_split(xp, y, test_size=0.2, random_state=1234)

# 2. 모델
model = make_pipeline(MinMaxScaler(), RandomForestClassifier())
model.fit(x_train, y_train)
print('폴리스코어: ', model.score(x_test, y_test))

score = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')
print('폴리 CV: ', score)
print('폴리 CV n빵: ', np.mean(score))


# 그냥 스코어:  0.9444444444444444
# CV:  [0.96551724 1.         1.         0.96428571 1.        ]
# CV n빵:  0.9859605911330049
# -------------------------------------------------------------
# 폴리스코어:  0.8888888888888888
# 폴리 CV:  [0.96551724 1.         1.         0.96428571 1.        ]
# 폴리 CV n빵:  0.9859605911330049