import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, accuracy_score
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import math
import time

from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, StratifiedKFold,\
    HalvingRandomSearchCV, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer


# 1. 데이터
file = 'D:\study_data\_data\wine/winequality-white.csv'

data = pd.read_csv(file, index_col=None, header=0,sep=';')

print(data.head)

x = data.values[:,0:11]
y = data.values[:,11]
le = LabelEncoder()
y = le.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.85,shuffle=True,random_state=1234)

parameters_xgb = {
            'n_estimators':[100,200,300,400,500],
            'learning_rate':[1,0.3,0.5,0.4],
            'max_depth':[None,2,3,4,5,6,7,8,9,10],
            'gamma':[0,1,2,4],
            'min_child_weight':[1],
            'subsample':[1],
            'colsample_bytree':[0,0.1,0.2,0.3,0.5,0.7,1] ,
            'colsample_bylevel':[1],
            'colsample_bynode':[0,0.1,0.2,0.3,0.5,0.7,1],
            'alpha':[0,0.1,0.01,0.001,1,2,10],
            'lambda':[0,0.1,0.01,0.001,1,2,10]
              } 

parameters_rnf = {
    'n_estimators':[100,200,300,400,500],
    'max_depth':[None,2,3,4,5,6,7,8,9,10,11,12,13,14],
    'min_samples_leaf':[3,5,7,10,11,13],
    'min_samples_split':[2,3,5,7,10],
    'n_jobs':[-1]
}


# 2. 모델
xgb = XGBClassifier(tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=0)
rnf = RandomForestClassifier(random_state=1234)

# model = xgb
model = rnf

# HRS = GridSearchCV(xgb, parameters_xgb, cv=6, n_jobs=-1, verbose=2)
# model = RandomizedSearchCV(xgb,  parameters_rnf, cv=5, n_jobs=-1, verbose=2)

# model = make_pipeline(MinMaxScaler(), xgb)
# model = make_pipeline(MinMaxScaler(), rnf)


# 3. 컴파일, 훈련
# model.fit(x_train, y_train, early_stopping_rounds=40, 
#           eval_set=[(x_train, y_train), (x_test, y_test)], eval_metric=['mlogloss'])

model.fit(x_train, y_train)


# 4. 평가, 예측
score = model.score(x_test, y_test)
print(score)
# print(HRS.best_params_)

# 0.6952380952380952 그냥 랜포
# 0.7047619047619048 그냥 랜포