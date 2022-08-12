import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from xgboost import XGBClassifier, XGBRegressor
import time
from sklearn.metrics import accuracy_score, r2_score
from sklearn.feature_selection import SelectFromModel

# 1. 데이터
datasets = load_digits()
x = datasets.data
y = datasets.target
le = LabelEncoder()
y = le.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=1234, train_size=0.8)

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
          )

print('테스트 스코어: ', model.score(x_test, y_test))

score = accuracy_score(y_test, model.predict(x_test))
print('acc_score 결과: ', score)

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


# 드랍 후 테스트 스코어:  0.9722222222222222
# 드랍 후 acc_score 결과:  0.9722222222222222

# 테스트 스코어:  0.9638888888888889
# acc_score 결과:  0.9638888888888889
# [0.         0.01065247 0.00948314 0.0035076  0.00445116 0.03638005
#  0.00443837 0.         0.         0.00218903 0.03074838 0.00059684
#  0.00161971 0.0335904  0.01468154 0.         0.         0.00158149
#  0.00918587 0.04863956 0.01072436 0.04088914 0.00139157 0.02727327
#  0.         0.00239637 0.0402486  0.00708215 0.01814266 0.04750339
#  0.05919282 0.         0.         0.09873619 0.01948814 0.01332176
#  0.00381819 0.02032899 0.038459   0.         0.         0.0149009
#  0.00687949 0.01813629 0.00848222 0.00592984 0.05046677 0.
#  0.         0.00202224 0.00536231 0.03339408 0.00206126 0.00688344
#  0.00250292 0.         0.         0.         0.01201635 0.00700372
#  0.11105001 0.00364167 0.01470325 0.03382109]
# -----------------------------------------------
# (1437, 48) (1437, 48)
# Thresh=0.001, n=48, R2: 97.78% 
