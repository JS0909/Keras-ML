from json import load
from bayes_opt import BayesianOptimization
from xgboost import XGBClassifier, XGBRegressor
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

# 1. 데이터
datasets = load_breast_cancer()
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
    'colsample_bytree' : (0.8, 1),
    'max_depth' : (10,13),
    'min_child_weight' : (20, 30),
    'reg_alpha' : (15, 25),
    'reg_lambda' : (0.1, 0.5),
    'subsample' : (0.7, 1)
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

# {'target': 0.956140350877193, 'params': {'colsample_bytree': 0.9903820991923078, 'max_depth': 12.848297385848632, 
#                                          'min_child_weight': 24.565663172733686, 'reg_alpha': 19.611954734525586, 
#                                          'reg_lambda': 0.34383483813471855, 'subsample': 0.8645248536920208}}

# {'target': 0.956140350877193, 'params': {'colsample_bytree': 0.9392938371195724, 'max_depth': 10.858418004851139, 
#                                          'min_child_weight': 22.26851453564203, 'reg_alpha': 20.513147690828912, 
#                                          'reg_lambda': 0.3877875879142253, 'subsample': 0.8269319380373382}}



model = XGBClassifier(n_estimators = 500, learning_rate= 0.02, colsample_bytree =max(min(0.9392938371195724,1),0) , max_depth=int(round(10.858418004851139)), min_child_weight =int(round(22.26851453564203)),
                      reg_alpha= max(20.513147690828912,0), reg_lambda=max(0.3877875879142253,0), subsample=max(min(0.8269319380373382,1),0))

model.fit(x_train, y_train)
y_pred = model.predict(x_test)
score = accuracy_score(y_test, y_pred)
print(score)

# 0.956140350877193