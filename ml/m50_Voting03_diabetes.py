import numpy as np
import pandas as pd

from sklearn.ensemble import VotingRegressor, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# 1. 데이터
dataset = load_breast_cancer()

x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target, train_size=0.8, random_state=704)

scl = MinMaxScaler()
x_train = scl.fit_transform(x_train)
x_test = scl.fit_transform(x_test)



parameters_rnf = {'n_estimators':[400],'max_depth':[None],'min_samples_leaf':[1],'min_samples_split':[2],
}

# 2. 모델
xg = XGBRegressor(n_estimators=100, learning_rate=1, max_depth=2, gamma=0, min_child_weight=1, subsample=1, colsample_bytree=0.5, 
                   colsample_bylevel=1, colsample_bynode=1, reg_alpha=0.01,
                   tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=0, random_state=1234,)
lg = LGBMRegressor()
cat = CatBoostRegressor(verbose=0)
rf = RandomForestRegressor(n_estimators=400, max_depth=None, min_samples_leaf=1 ,min_samples_split=2)

model = VotingRegressor(estimators=[('XG', xg), ('LG', lg), ('CAT', cat), ('RF', rf)],
                        #  voting='hard'
                         )

# 3. 훈련
model.fit(x_train,y_train)

# 4. 평가, 예측
y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print('voting result: ', round(r2, 4))



classifiers = [xg, lg, cat, rf]
for model2 in classifiers:
    model2.fit(x_train, y_train)
    y_pred = model2.predict(x_test)
    r22 = r2_score(y_test, y_pred)
    class_name = model2.__class__.__name__
    print('{0} 정확도: {1:.4f}'.format(class_name, r22))
    

# voting result:  0.7741
# XGBRegressor 정확도: 0.3273
# LGBMRegressor 정확도: 0.7761
# CatBoostRegressor 정확도: 0.8222
# RandomForestRegressor 정확도: 0.7374