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
import glob

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

train_input = pd.read_csv(train_input_path+'CASE_01.csv')
for i in range(58):
    if i==0:
        a=i+2
    else:
        a=i+1
    a=str(a)
    if i <=8:
        train_input = pd.concat([pd.read_csv(train_input_path+'CASE_0'+a+'.csv'), train_input],axis=0)
    else:
        train_input = pd.concat([pd.read_csv(train_input_path+'CASE_'+a+'.csv'), train_input], axis=0)


test_input = pd.read_csv(test_input_path+'TEST_01.csv')
for i in range(6):
    if i==0:
        a=i+2
    else:
        a=i+1
    a=str(a)
    if i <=8:
        test_input = pd.concat([pd.read_csv(test_input_path+'TEST_0'+a+'.csv'), test_input],axis=0)
    else:
        test_input = pd.concat([pd.read_csv(test_input_path+'TEST_'+a+'.csv'),test_input], axis=0)
      
  
train_target = pd.read_csv(train_target_path+'CASE_01.csv')
for i in range(58):
    if i==0:
        a=i+2
    else:
        a=i+1
    a=str(a)
    if i <=8:
        train_target = pd.concat([pd.read_csv(train_target_path+'CASE_0'+a+'.csv'), train_target],axis=0)
    else:
        train_target = pd.concat([pd.read_csv(train_target_path+'CASE_'+a+'.csv'), train_target],axis=0)


test_target = pd.read_csv(test_target_path+'TEST_01.csv')
for i in range(6):
    if i==0:
        a=i+2
    else:
        a=i+1
    a=str(a)
    if i <=8:
        test_target = pd.concat([pd.read_csv(test_target_path+'TEST_0'+a+'.csv'), test_target],axis=0)
    else:
        test_target = pd.concat([pd.read_csv(test_target_path+'TEST_'+a+'.csv'), test_target], axis=0)
        
        
# print(train_input.head)
# print(test_input.head)
# print(train_target.head)
# print(test_target.head)

print(np.array(train_input).shape) # (2653267, 43)
print(np.array(test_input).shape) # (335520, 42)
