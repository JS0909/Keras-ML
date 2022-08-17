import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, accuracy_score
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import math
import joblib


from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, StratifiedKFold,\
    HalvingRandomSearchCV, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from imblearn.over_sampling import SMOTE

# 1. 데이터
file = 'D:\study_data\_data\dacon_vegi/'
train_input_path = file+'train_input/'
test_input_path = file+'test_input/'
train_target_path = file+'train_target/'
test_target_path = file+'test_target/'

train_x = pd.read_csv(train_input_path+'CASE_01.csv', index_col=False)
train_y = pd.read_csv(train_target_path+'CASE_01.csv', index_col=False)

train_x = train_x.drop(['시간'], axis=1)
train_y = train_y.drop(['시간'], axis=1)

train_x_arr = train_x.values
# train_x1 = train_x_arr[:1440][:]
train_y_arr =train_y.values
print(train_x_arr.shape) # (41760, 37)

xs = []
for i in range(len(train_x_arr)):
    a = train_x_arr[i:1440+i][:]
    i+=1440+i
    xs.append(a)

print(xs)
# print(xs.shape) # (41760,)
print(xs[0][:].shape) # (1440, 37)

print(train_y.shape) # (29, 1)

new_y = []
for a in range(len(train_y)):
    for i in range(1440):
        new_y.append(train_y_arr[a][0])

print(new_y)

# 2. 모델
model = RandomForestRegressor()






# 3. 컴파일, 훈련
model.fit(xs[0][:], train_y[0][1])

# 4. 평가, 예측
print()

