from bayes_opt import BayesianOptimization
from xgboost import XGBClassifier, XGBRegressor
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

# 1. 데이터
datasets = fetch_california_housing()
x, y = datasets.data, datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=134, train_size=0.8)

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
    'colsample_bytree' : (0.3, 0.9),
    'max_depth' : (5,15),
    'min_child_weight' : (15, 25),
    'reg_alpha' : (0.01, 1),
    'reg_lambda' : (0.01, 1),
    'subsample' : (0.5, 1.5)
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
    model = XGBRegressor(**params)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    score = r2_score(y_test, y_pred)
    
    return score

lgb_bo = BayesianOptimization(f=lgb_function, pbounds=bayseian_params, random_state=123)

lgb_bo.maximize(init_points=3, n_iter=50)
print(lgb_bo.max)

# {'target': 0.8466475775331651, 'params': {'colsample_bytree': 0.65311540647911, 'max_depth': 10.816414070677556, 
#                                           'min_child_weight': 21.02753480389905, 'reg_alpha': 0.2862378674095809, 
#                                           'reg_lambda': 0.7080594884897791, 'subsample': 0.9730104828479126}}

# {'target': 0.8465945485898457, 'params': {'colsample_bytree': 0.7330057357620108, 'max_depth': 13.783683510607842, 
#                                           'min_child_weight': 15.034221970820731, 'reg_alpha': 0.9316372938750783, 
#                                           'reg_lambda': 0.8363767216896922, 'subsample': 1.407362530421373}}

model = XGBRegressor(n_estimators = 500, learning_rate= 0.02, colsample_bytree =max(min(0.65311540647911,1),0) ,
                     max_depth=int(round(10.816414070677556)), min_child_weight =int(round(21.02753480389905)),
                      reg_alpha= max(0.2862378674095809,0), reg_lambda=max(0.7080594884897791,0), subsample=max(min(0.9730104828479126,1),0))

model.fit(x_train, y_train)
y_pred = model.predict(x_test)
score = r2_score(y_test, y_pred)
print(score)

# 0.8466475775331651