import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import time
import warnings
warnings.filterwarnings('ignore') # warnig 출력 안함

from bayes_opt import BayesianOptimization
from sklearn.model_selection import train_test_split, KFold

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import r2_score, accuracy_score
from xgboost import XGBClassifier



# 1. 데이터
path = 'D:\study_data\_data\kaggle_titanic/'
train_set = pd.read_csv(path+'train.csv')
test_set = pd.read_csv(path+'test.csv')

train_set = train_set.drop(columns='Cabin', axis=1)
train_set['Age'].fillna(train_set['Age'].mean(), inplace=True)
print(train_set['Embarked'].mode())
train_set['Embarked'].fillna(train_set['Embarked'].mode()[0], inplace=True)
train_set.replace({'Sex':{'male':0,'female':1}, 'Embarked':{'S':0,'C':1,'Q':2}}, inplace=True)

# train_set 불러올 때와 마찬가지로 전처리시켜야 model.predict에 넣어서 y값 구하기가 가능함-----------
print(test_set.isnull().sum())
test_set = test_set.drop(columns='Cabin', axis=1)
test_set['Age'].fillna(test_set['Age'].mean(), inplace=True)
test_set['Fare'].fillna(test_set['Fare'].mean(), inplace=True)
print(test_set['Embarked'].mode())
test_set['Embarked'].fillna(test_set['Embarked'].mode()[0], inplace=True)
test_set.replace({'Sex':{'male':0,'female':1}, 'Embarked':{'S':0,'C':1,'Q':2}}, inplace=True)
test_set = test_set.drop(columns = ['PassengerId','Name','Ticket'],axis=1)
#---------------------------------------------------------------------------------------------------

y = train_set['Survived']
x = train_set.drop(columns = ['PassengerId','Name','Ticket','Survived'],axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=134, train_size=0.8, stratify=y)

scl = StandardScaler()
x_train = scl.fit_transform(x_train)
x_test = scl.transform(x_test)

# 2. 모델
# bayseian_params = {
#     'colsample_bytree' : (0.5, 1),
#     'max_depth' : (6,16),
#     'min_child_weight' : (1, 50),
#     'reg_alpha' : (0.01, 50),
#     'reg_lambda' : (0.001, 1),
#     'subsample' : (0.5, 1)
# }

bayseian_params = {
    'colsample_bytree' : (0.6, 1),
    'max_depth' : (12,17),
    'min_child_weight' : (1, 6),
    'reg_alpha' : (40, 51),
    'reg_lambda' : (0.5, 1.5),
    'subsample' : (0.1, 1.5)
}


def lgb_function(max_depth, min_child_weight,subsample, colsample_bytree, reg_lambda,reg_alpha):
    params ={
        'n_estimators' : 500, 'learning_rate' : 0.02,
        'max_depth' : int(round(max_depth)),                    # 정수만
        'min_child_weight' : int(round(min_child_weight)),
        'subsample' : max(min(subsample,1),0),                  # 0~1 사이값만
        'colsample_bytree' : max(min(colsample_bytree,1),0),
        'reg_lambda' : max(reg_lambda,0),                       # 양수만
        'reg_alpha' : max(reg_alpha,0),
    }
    
    # *여러개의인자를받겠다
    # **키워드받겠다(딕셔너리형태)
    model = XGBClassifier(**params)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    score = accuracy_score(y_test, y_pred)
    
    return score

lgb_bo = BayesianOptimization(f=lgb_function, pbounds=bayseian_params, random_state=123)

lgb_bo.maximize(init_points=3, n_iter=50)
print(lgb_bo.max)

# {'target': 0.8491620111731844, 'params': {'colsample_bytree': 0.8608381131350089, 'max_depth': 14.783683510607842, 
#                                           'min_child_weight': 1.1676876570215784, 'reg_alpha': 46.548028606884, 
#                                           'reg_lambda': 0.8348892373414167, 'subsample': 0.9536812652106865}}

# {'target': 0.8547486033519553, 'params': {'colsample_bytree': 1.0, 'max_depth': 14.60459381134441, 
#                                           'min_child_weight': 1.0, 'reg_alpha': 44.704038293798945, 
#                                           'reg_lambda': 1.5, 'subsample': 1.306762686709858}}

model = XGBClassifier(n_estimators = 500, learning_rate= 0.02, colsample_bytree =max(min(1.0,1),0) , max_depth=int(round(14.60459381134441)), min_child_weight =int(round(1.0)),
                      reg_alpha= max(44.704038293798945,0), reg_lambda=max(1.5,0), subsample=max(min(1.306762686709858,1),0))

model.fit(x_train, y_train)
y_pred = model.predict(x_test)
score = accuracy_score(y_test, y_pred)
print(score)

# 0.8547486033519553