from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, MinMaxScaler
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline

# 1. 데이터
datasets = load_digits()
x,y = datasets.data, datasets.target
print(x.shape, y.shape) # (1797, 64) (1797,)

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
print(xp.shape) # (1797, 2144)

x_train, x_test, y_train, y_test = train_test_split(xp, y, test_size=0.2, random_state=1234)

# 2. 모델
model = make_pipeline(MinMaxScaler(), RandomForestClassifier())
model.fit(x_train, y_train)
print('폴리스코어: ', model.score(x_test, y_test))

score = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')
print('폴리 CV: ', score)
print('폴리 CV n빵: ', np.mean(score))


# 그냥 스코어:  0.975
# CV:  [0.97916667 0.98611111 0.97212544 0.95470383 0.96515679]
# CV n빵:  0.9714527680991096
# -------------------------------------------------------------
# 폴리스코어:  0.9833333333333333
# 폴리 CV:  [0.96180556 0.98611111 0.96864111 0.95818815 0.96167247]
# 폴리 CV n빵:  0.9672836817653889