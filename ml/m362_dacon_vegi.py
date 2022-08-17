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

'''
train_input = pd.read_csv(train_input_path+'CASE_01.csv')
for i in range(58):
    if i==0:
        a=i+2
    else:
        a=i+1
    a=str(a)
    if i <=8:
        train_input = pd.concat([train_input, pd.read_csv(train_input_path+'CASE_0'+a+'.csv')],axis=0)
    else:
        train_input = pd.concat([train_input, pd.read_csv(train_input_path+'CASE_'+a+'.csv')], axis=0)


test_input = pd.read_csv(test_input_path+'TEST_01.csv')
for i in range(6):
    if i==0:
        a=i+2
    else:
        a=i+1
    a=str(a)
    if i <=8:
        test_input = pd.concat([test_input, pd.read_csv(test_input_path+'TEST_0'+a+'.csv')],axis=0)
    else:
        test_input = pd.concat([test_input, pd.read_csv(test_input_path+'TEST_'+a+'.csv')], axis=0)
      
  
train_target = pd.read_csv(train_target_path+'CASE_01.csv')
for i in range(58):
    if i==0:
        a=i+2
    else:
        a=i+1
    a=str(a)
    if i <=8:
        train_target = pd.concat([train_target, pd.read_csv(train_target_path+'CASE_0'+a+'.csv')],axis=0)
    else:
        train_target = pd.concat([train_target, pd.read_csv(train_target_path+'CASE_'+a+'.csv')],axis=0)


test_target = pd.read_csv(test_target_path+'TEST_01.csv')
for i in range(6):
    if i==0:
        a=i+2
    else:
        a=i+1
    a=str(a)
    if i <=8:
        test_target = pd.concat([test_target, pd.read_csv(test_target_path+'TEST_0'+a+'.csv')],axis=0)
    else:
        test_target = pd.concat([test_target, pd.read_csv(test_target_path+'TEST_'+a+'.csv')], axis=0)
'''   

# joblib.dump(train_input, file+'datasets/train_input.dat')
# joblib.dump(train_target, file+'datasets/train_target.dat')
# joblib.dump(test_input, file+'datasets/test_input.dat')
# joblib.dump(test_target, file+'datasets/test_target.dat')

train_input = joblib.load(file+'datasets/train_input.dat')
train_target = joblib.load(file+'datasets/train_target.dat')
test_input = joblib.load(file+'datasets/test_input.dat')
test_target = joblib.load(file+'datasets/test_target.dat')

# print(train_input.head)
# print(test_input.head)
# print(train_target.head)
# print(test_target.head)

# print(np.array(train_input).shape) # (2653267, 43)
# print(np.array(test_input).shape) # (335520, 42)

# print(train_input.describe())
# print(train_input.info())
# print(train_input.isnull().sum())

# print(test_input.describe())
# print(test_input.info())
# print(test_input.isnull().sum())

# 24시간 데이터로 다음날 0시의 잎 증감률을 예측
# 하루 = 1440 행

train_input=train_input.drop(['시간'], axis=1)

imp = IterativeImputer()
train_input = imp.fit_transform(train_input)
train_target = imp.fit_transform(train_target)


train_input_arr = train_input.values
train_input_arr = train_input_arr[1:1440][:]
print(train_input_arr)


# 2. 모델




# 3. 컴파일, 훈련


# 4. 평가, 예측

