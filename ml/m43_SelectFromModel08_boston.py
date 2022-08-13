import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from xgboost import XGBClassifier, XGBRegressor
import time
from sklearn.metrics import accuracy_score, r2_score
from sklearn.feature_selection import SelectFromModel

# 1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

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
    
    selection_model = XGBRegressor(n_estimators=100,
              learning_rate=1,
              max_depth=2,
              gamma=0,
              min_child_weight=1,
              subsample=1,
              colsample_bytree=0.5,
              colsample_bylevel=1,
              reg_alpha=0.01,
              colsample_bynode=1,
              tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=0, random_state=1234,
              )
    
    selection_model.fit(select_x_train, y_train)
    
    y_predict = selection_model.predict(select_x_test)
    score = r2_score(y_test, y_predict)
    print('Thresh=%.3f, n=%d, R2: %.2f%%'%(thresholds[i], select_x_train.shape[1], score*100), '\n')

    if score >= bscore:
        bscore = score
        idx_=i

f_to_drop = []
for i in range(len(thresholds)):
    if thresholds[idx_]>=thresholds[i]:
        f_to_drop.append(i)
        
print(f_to_drop)

xaf_train = np.delete(x_train, f_to_drop, axis=1)
xaf_test = np.delete(x_test, f_to_drop, axis=1)

model.fit(xaf_train, y_train)

print('드랍 후 테스트 스코어: ', model.score(xaf_test, y_test))

score = r2_score(y_test, model.predict(xaf_test))
print('드랍 후 score 결과: ', score)




# 테스트 스코어:  0.7672683672355034
# acc_score 결과:  0.7672683672355034
# [0.05859446 0.         0.05091146 0.02294821 0.01811893 0.117872
#  0.03080538 0.01218672 0.02632577 0.06195151 0.05387018 0.08220655
#  0.4642088 ]
# -----------------------------------------------
# (404, 8) (404, 8)
# Thresh=0.031, n=8, R2: 87.92% 

# 드랍 후 테스트 스코어:  0.7824965773283676
# 드랍 후 score 결과:  0.7824965773283676