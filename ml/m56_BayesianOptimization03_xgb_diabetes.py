from json import load
from bayes_opt import BayesianOptimization
from xgboost import XGBClassifier, XGBRegressor
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

# 1. 데이터
datasets = load_diabetes()
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
    'colsample_bytree' : (0.5, 0.7),
    'max_depth' : (10,18),
    'min_child_weight' : (30, 35),
    'reg_alpha' : (43, 47),
    'reg_lambda' : (0.001, 0.01),
    'subsample' : (0.4, 0.7)
}

# {'target': 0.4154637633888485, 'params': {'colsample_bytree': 0.5, 'max_depth': 16.0, 'min_child_weight': 32.34096686624792, 
#                                           'reg_alpha': 45.94945765955618, 'reg_lambda': 0.001, 'subsample': 0.5}}

# {'target': 0.4142491922910905, 'params': {'colsample_bytree': 0.502542868427671, 'max_depth': 17.73652974881533, 'min_child_weight': 32.42438850399354, 
#                                           'reg_alpha': 43.84144259300621, 'reg_lambda': 0.00949494858720348, 'subsample': 0.5169589004547956}}

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

model = XGBRegressor(n_estimators = 500, learning_rate= 0.02, colsample_bytree =max(min(0.5,1),0) ,
                     max_depth=int(round(16.0)), min_child_weight =int(round(32.34096686624792)),
                      reg_alpha= max(45.94945765955618,0), reg_lambda=max(0.001,0), subsample=max(min(0.5169589004547956,1),0))

model.fit(x_train, y_train)
y_pred = model.predict(x_test)
score = r2_score(y_test, y_pred)
print(score)

# 0.4118576361727567