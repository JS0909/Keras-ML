import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
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
# model = XGBRegressor(n_estimators=100,
#               learning_rate=1,
#               max_depth=2,
#               gamma=0,
#               min_child_weight=1,
#               subsample=1,
#               colsample_bytree=0.5,
#               colsample_bylevel=1,
#               colsample_bynode=1,
#               reg_alpha=0.01,
#               reg_lambd=1,
#               tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=0, random_state=1234,
#               )
model = LinearRegression()
model.fit(x_train, y_train)

print('테스트 스코어: ', model.score(x_test, y_test))

r2 = r2_score(y_test, model.predict(x_test))
print('acc_score 결과: ', r2)

print(model.feature_importances_)
# [0.02737029 0.06044502 0.2727516  0.07338018 0.02401855 0.06909694
#  0.03971948 0.24999845 0.07974161 0.10347791]

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


# (353, 9) (353, 9)
# (353, 7) (353, 7)
# (353, 1) (353, 1)
# (353, 5) (353, 5)
# (353, 10) (353, 10)
# (353, 6) (353, 6)
# (353, 8) (353, 8)
# (353, 2) (353, 2)
# (353, 4) (353, 4)
# (353, 3) (353, 3)
# 칼럼 갯수 중복 없음
# 해당 피처임포턴스보다 작은 놈을 빼버린다 / 자기와 자기보다 큰 임포턴스의 피처만 남겨놓는다



# 최상의 매개변수:  {'colsample_bylevel': 1, 'colsample_bynode': 1, 'colsample_bytree': 0.5, 'gamma': 0, 'learning_rate': 1, 'max_depth': 2, 'min_child_weight': 1, 'n_estimators': 100, 'reg_alpha': 0.01, 'reg_lambda': 1, 'subsample': 1}
# 최상의 점수:  0.9824175824175825
# 테스트 스코어:  0.9649122807017544

# n_estimators = 100
# 테스트 스코어:  0.9912280701754386
