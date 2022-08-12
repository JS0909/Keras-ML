import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from xgboost import XGBClassifier, XGBRegressor
import time
from sklearn.metrics import accuracy_score, r2_score
from sklearn.feature_selection import SelectFromModel

# 1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=1234, train_size=0.8, stratify=y)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델
model = XGBClassifier(n_estimators=100,
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
bscore = 0
idx_ = 0
for i in range(len(thresholds)):
    selection = SelectFromModel(model, threshold=thresholds[i], prefit=True)
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    print(select_x_train.shape, select_x_train.shape)
    
    selection_model = XGBClassifier(n_estimators=100,
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
    score = accuracy_score(y_test, y_predict)
    print('Thresh=%.3f, n=%d, R2: %.2f%%'%(thresholds[i], select_x_train.shape[1], score*100), '\n')

    if score >= bscore:
        bscore = score
        idx_=i

f_to_drop = []
for i in range(len(thresholds)):
    if thresholds[idx_]>=thresholds[i]:
        f_to_drop.append(i)
        
print(f_to_drop)
# [0, 7, 8, 11, 15, 16, 22, 24, 31, 32, 39, 40, 47, 48, 55, 56, 57]

xaf_train = np.delete(x_train, f_to_drop, axis=1)
xaf_test = np.delete(x_test, f_to_drop, axis=1)

model.fit(xaf_train, y_train, early_stopping_rounds=10, 
          eval_set=[(xaf_train, y_train), (xaf_test, y_test)],
          )

print('드랍 후 테스트 스코어: ', model.score(xaf_test, y_test))

score = accuracy_score(y_test, model.predict(xaf_test))
print('드랍 후 acc_score 결과: ', score)


# 테스트 스코어:  0.9649122807017544
# acc_score 결과:  0.8492063492063493
# [0.         0.07496004 0.         0.00676468 0.00306201 0.01581421
#  0.00145857 0.08026031 0.         0.00350958 0.02287502 0.
#  0.         0.00524435 0.00077113 0.00744511 0.00061739 0.00242535
#  0.00474947 0.00149277 0.00549055 0.00765657 0.32585773 0.34295195
#  0.02784067 0.         0.04727238 0.00849394 0.00108742 0.0018987 ]
# -----------------------------------------------

# (455, 14) (455, 14)
# Thresh=0.005, n=14, R2: 92.46% 

# (455, 13) (455, 13)
# Thresh=0.005, n=13, R2: 92.46% 

# 드랍 후 테스트 스코어:  0.9736842105263158
# 드랍 후 acc_score 결과:  0.9736842105263158