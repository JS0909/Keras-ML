import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from xgboost import XGBClassifier, XGBRegressor
import time
from sklearn.metrics import accuracy_score, r2_score
from sklearn.feature_selection import SelectFromModel

# 1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) #(442, 10) (442,)

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=1234, train_size=0.8)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델
model = XGBRegressor(n_estimators=100,
              learning_rate=1,
              max_depth=2,
              gamma=0,
              min_child_weight=1,
              subsample=1,
              colsample_bytree=0.5,
              colsample_bylevel=1,
              colsample_bynode=1,
              reg_alpha=0.01,
              reg_lambd=1,
              tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=0, random_state=1234,
              )

model.fit(x_train, y_train, early_stopping_rounds=10, 
          eval_set=[(x_train, y_train), (x_test, y_test)],
          eval_metric=['logloss'],
          )

print('테스트 스코어: ', model.score(x_test, y_test))

r2 = r2_score(y_test, model.predict(x_test))
print('acc_score 결과: ', r2)

print(model.feature_importances_)

thresholds = model.feature_importances_
print('-----------------------------------------------')
for thresh in thresholds:
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    print(select_x_train.shape, select_x_train.shape)
    
    selection_model = XGBRegressor(n_estimators=100,
              learning_rate=1,
              max_depth=2,
              gamma=0,
              min_child_weight=1,
              subsample=1,
              colsample_bytree=0.5,
              colsample_bylevel=1,
              colsample_bynode=1,
              reg_alpha=0.01,
              tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=0, random_state=1234,
              )
    
    selection_model.fit(select_x_train, y_train)
    
    y_predict = selection_model.predict(select_x_test)
    score = r2_score(y_test, y_predict)
    print('Thresh=%.3f, n=%d, R2: %.2f%%'%(thresh, select_x_train.shape[1], score*100), '\n')


# 테스트 스코어:  0.34273103377168257
# acc_score 결과:  0.34273103377168257
# [0.02737029 0.06044502 0.2727516  0.07338018 0.02401855 0.06909694
#  0.03971948 0.24999845 0.07974161 0.10347791]

# Thresh=0.040, n=8, R2: 19.73%


# 최상의 매개변수:  {'colsample_bylevel': 1, 'colsample_bynode': 1, 'colsample_bytree': 0.5, 'gamma': 0, 'learning_rate': 1, 'max_depth': 2, 'min_child_weight': 1, 'n_estimators': 100, 'reg_alpha': 0.01, 'reg_lambda': 1, 'subsample': 1}
# 최상의 점수:  0.9824175824175825
# 테스트 스코어:  0.9649122807017544

# n_estimators = 100
# 테스트 스코어:  0.9912280701754386


# 테스트 스코어:  0.7770453430634321
# acc_score 결과:  0.7770453430634321
# [0.05857961 0.00393236 0.00292832 0.00877082 0.00469158 0.00719566
#  0.00570423 0.01011029 0.00493441 0.00768254 0.0177074  0.00923314
#  0.01927559 0.13997956 0.00124716 0.02897144 0.01076196 0.04513867
#  0.00201243 0.02087582 0.00040969 0.00409521 0.00405243 0.02482739
#  0.00253851 0.13731663 0.00833687 0.00127003 0.00020432 0.00492075
#  0.00903463 0.00481056 0.00323922 0.00532154 0.00669941 0.02376849
#  0.02257365 0.01172577 0.00438431 0.00242111 0.00433011 0.00131364
#  0.01061864 0.01473258 0.00594154 0.00912848 0.00475747 0.00406218
#  0.01408234 0.00117094 0.01585765 0.14115755 0.0651608  0.01600262]
# -----------------------------------------------
# (464809, 47) (464809, 47)
# Thresh=0.002, n=47, R2: 77.97% 
