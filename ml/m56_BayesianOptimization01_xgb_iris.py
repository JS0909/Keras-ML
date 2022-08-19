from json import load
from bayes_opt import BayesianOptimization
from xgboost import XGBClassifier, XGBRegressor
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

# 1. 데이터
datasets = load_iris()
x, y = datasets.data, datasets.target

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
    'colsample_bytree' : (0.5, 0.7),
    'max_depth' : (7,13),
    'min_child_weight' : (1, 2),
    'reg_alpha' : (10, 25),
    'reg_lambda' : (0.001, 1),
    'subsample' : (0.5, 0.7)
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

# {'target': 1.0, 'params': {'colsample_bytree': 0.5, 
#                            'max_depth': 9.171285076955737, 
#                            'min_child_weight': 1.0, 
#                            'reg_alpha': 15.031619680955929, 'reg_lambda': 0.001, 'subsample': 0.5}}

# {'target': 1.0, 'params': {'colsample_bytree': 0.6392938371195723, 'max_depth': 8.716836009702277, 
#                            'min_child_weight': 1.226851453564203, 'reg_alpha': 18.269721536243367, 
#                            'reg_lambda': 0.7197495008157775, 'subsample': 0.5846212920248922}}

model = XGBClassifier(n_estimators = 500, learning_rate= 0.02, colsample_bytree =max(min(0.6392938371195723,1),0) , max_depth=int(round(8.716836009702277)), min_child_weight =int(round(1.226851453564203)),
                      reg_alpha= max(18.269721536243367,0), reg_lambda=max(0.7197495008157775,0), subsample=max(min(0.5846212920248922,1),0))

model.fit(x_train, y_train)
y_pred = model.predict(x_test)
score = accuracy_score(y_test, y_pred)
print(score)

# 1.0