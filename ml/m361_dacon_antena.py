import pandas as pd
import random
import os
import numpy as np
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
seed_everything(99) # Seed 고정

filepath = 'D:/study_data/_data/dacon_antena/'

train = pd.read_csv(filepath + 'train.csv',index_col=0)
test = pd.read_csv(filepath + 'test.csv').drop(columns=['ID'])


train_x = train.filter(regex='X') # Input : X Featrue
train_y = train.filter(regex='Y') # Output : Y Feature

cols = ["X_10","X_11"]
train[cols] = train[cols].replace(0, np.nan)

# train[cols].fillna(train[cols].mean(), inplace=True)

# imp = KNNImputer()

imp = IterativeImputer(estimator = LinearRegression(), 
                       tol= 1e-10, 
                       max_iter=30, 
                       verbose=2, 
                       imputation_order='roman')

train = imp.fit_transform(train)


# model = MultiOutputRegressor(XGBRegressor(n_estimators=150, learning_rate=0.08, gamma = 1, subsample=0.75, colsample_bytree = 1, max_depth=7) )
model = MultiOutputRegressor(LinearRegression())
# model = RandomForestRegressor()

model.fit(train_x, train_y)
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

# 0.8708800720214304 랜덤포레스트, 멀티아웃풋 아니라 그런가;

# 0.039531560921963645 x10, x11 칼럼 결측치 처리(IterativeImputer)

# 0.039531560921963645 x10, x11 칼럼 결측치 처리(KNNImputer)