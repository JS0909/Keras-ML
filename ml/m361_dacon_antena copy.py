import pandas as pd
import random
import os
import numpy as np
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
seed_everything(704) # Seed 고정

filepath = 'D:/study_data/_data/dacon_antena/'

train = pd.read_csv(filepath + 'train.csv')
test = pd.read_csv(filepath + 'test.csv').drop(columns=['ID'])


train_x = train.filter(regex='X') # Input : X Featrue
train_y = train.filter(regex='Y') # Output : Y Feature

train_x = train_x.drop(['X_10', 'X_11'], axis=1)
test = test.drop(['X_10', 'X_11'], axis=1)

model = MultiOutputRegressor(XGBRegressor(n_estimators=100, learning_rate=0.08, gamma = 0, subsample=0.75, colsample_bytree = 1, max_depth=7) ).fit(train_x, train_y)
# model = MultiOutputRegressor(LinearRegression()).fit(train_x, train_y)
# model = MultiOutputRegressor(RandomForestRegressor()).fit(train_x, train_y)

preds = model.predict(test)
print(model.score(train_x, train_y))

submit = pd.read_csv(filepath +'sample_submission.csv')
for idx, col in enumerate(submit.columns):
    if col=='ID':
        continue
    submit[col] = preds[:,idx-1]

submit.to_csv(filepath + 'submission.csv', index=False)



# 0.2842927683724148 xg부스트 스렉이

# 0.03953156092196286 칼럼 드랍 없이 / 제출 446위

# 0.03932477616005312 x10, x11 칼럼 드랍 리니어