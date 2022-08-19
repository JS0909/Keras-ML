from json import load
from bayes_opt import BayesianOptimization
from lightgbm import LGBMRegressor, LGBMClassifier
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

# 1. 데이터
datasets = load_breast_cancer()
x, y = datasets.data, datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=134, train_size=0.8)

scl = StandardScaler()
x_train = scl.fit_transform(x_train)
x_test = scl.transform(x_test)

# 2. 모델
bayseian_params = {
    'max_depth' : (6,16), # bayseian의 파라미터는 범위값으로 지정해서 준다
    'num_leaves' : (24, 64),
    'min_child_samples' : (10, 200),
    'min_child_weight' : (1, 50),
    'subsample' : (0.5, 1),
    'colsample_bytree' : (0.5, 1),
    'max_bin' : (10, 500),
    'reg_lambda' : (0.001, 10),
    'reg_alpha' : (0.01, 50)
}

def lgb_function(max_depth, num_leaves,min_child_samples,min_child_weight,subsample, colsample_bytree, max_bin,reg_lambda,reg_alpha):
    params ={
        'n_estimators' : 500, 'learning_rate' : 0.02,
        'max_depth' : int(round(max_depth)),                    # 정수만
        'num_leaves' : int(round(num_leaves)),
        'min_child_samples' : int(round(min_child_samples)),
        'min_child_weight' : int(round(min_child_weight)),
        'subsample' : max(min(subsample,1),0),                  # 0~1 사이값만
        'colsample_bytree' : max(min(colsample_bytree,1),0),
        'max_bin' : max(int(round(max_bin)),10),                # 10 이상의 정수만
        'reg_lambda' : max(reg_lambda,0),                       # 양수만
        'reg_alpha' : max(reg_alpha,0),
    }
    
    # *여러개의인자를받겠다
    # **키워드받겠다(딕셔너리형태)
    model = LGBMClassifier(**params)
    model.fit(x_train, y_train, eval_set=[(x_train,y_train), (x_test, y_test)],eval_metric='accuracy',
              verbose=0, early_stopping_rounds=50)
    y_pred = model.predict(x_test)
    score = accuracy_score(y_test, y_pred)
    
    return score

lgb_bo = BayesianOptimization(f=lgb_function, pbounds=bayseian_params, random_state=134)

lgb_bo.maximize(init_points=5, n_iter=100)
print(lgb_bo.max)

# {'target': 0.956140350877193, 'params': {'colsample_bytree': 0.6410534072751939, 'max_bin': 36.78575645389972, 
#                                          'max_depth': 11.009510743761751, 'min_child_samples': 35.68790676289743, 
#                                          'min_child_weight': 7.49696378644739, 'num_leaves': 34.445503029140994, 
#                                          'reg_alpha': 2.837809768308744, 'reg_lambda': 6.925944311472341, 'subsample': 0.6685549538143083}}