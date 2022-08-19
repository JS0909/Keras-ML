from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, RobustScaler, StandardScaler,\
    QuantileTransformer, PowerTransformer # = 이상치에 자유로운 편

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


# 1. 데이터
datasets = load_diabetes()
x,y = datasets.data, datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)

model = RandomForestRegressor()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
result = r2_score(y_test, y_pred)
print('no scaler: ', round(result,4))



sclist = [StandardScaler(),MinMaxScaler(),MaxAbsScaler(),RobustScaler(),QuantileTransformer(),
          PowerTransformer(method='yeo-johnson'), # 디폴트 
          PowerTransformer(method='box-cox')
            ]
               
for scl in sclist:

    if str(scl) == str(PowerTransformer(method='box-cox')):
        try:
            x_train = scl.fit_transform(x_train)
        except:
            print('box-cox 안됨')
            break
          
    x_train = scl.fit_transform(x_train)
    x_test = scl.transform(x_test)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    result = r2_score(y_test, y_pred)
    print(str(scl).strip('()')+'결과: ', round(result,4))
    


# no scaler:  0.402
# StandardScaler결과:  0.4417
# MinMaxScaler결과:  0.4139
# MaxAbsScaler결과:  0.4547
# RobustScaler결과:  0.3976
# QuantileTransformer결과:  0.4044
# PowerTransformer결과:  0.4042