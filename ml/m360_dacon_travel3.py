import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, accuracy_score
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import math
import time

from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, StratifiedKFold,\
    HalvingRandomSearchCV, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer


# 1. 데이터
filepath = 'D:\study_data\_data\dacon_travel/'
train = pd.read_csv(filepath+'train.csv', index_col=0)
test = pd.read_csv(filepath+'test.csv', index_col=0)

# print(train.head())
# print(train.info())
# print(train.isnull().sum())

# 결측치 TypeofContact 빼고 중간값으로 대체함, 데이터 수치들 보면 중간값이 제일 무난할거 같음--------------------
train['Age'].fillna(train['Age'].median(), inplace=True)
train['TypeofContact'].fillna('N', inplace=True) # N으로 채운 이유: 콘택 타입 없는 건 '없음'으로 주고 처리하기 위해
train['DurationOfPitch'].fillna(train['DurationOfPitch'].median(), inplace=True)
train['NumberOfFollowups'].fillna(train['NumberOfFollowups'].median(), inplace=True)
train['PreferredPropertyStar'].fillna(train['PreferredPropertyStar'].median(), inplace=True)
train['NumberOfTrips'].fillna(train['NumberOfTrips'].median(), inplace=True)
train['NumberOfChildrenVisiting'].fillna(train['NumberOfChildrenVisiting'].median(), inplace=True)
train['MonthlyIncome'].fillna(train['MonthlyIncome'].median(), inplace=True)

test['Age'].fillna(test['Age'].median(), inplace=True)
test['TypeofContact'].fillna('N', inplace=True)
test['DurationOfPitch'].fillna(test['DurationOfPitch'].median(), inplace=True)
test['NumberOfFollowups'].fillna(test['NumberOfFollowups'].median(), inplace=True)
test['PreferredPropertyStar'].fillna(test['PreferredPropertyStar'].median(), inplace=True)
test['NumberOfTrips'].fillna(test['NumberOfTrips'].median(), inplace=True)
test['NumberOfChildrenVisiting'].fillna(test['NumberOfChildrenVisiting'].median(), inplace=True)
test['MonthlyIncome'].fillna(test['MonthlyIncome'].median(), inplace=True)
# print(train.isnull().sum())
#-----------------------------------------------------------------------------------------------------------

# object타입 라벨인코딩--------------------
le = LabelEncoder()
idxarr = train.columns
idxarr = np.array(idxarr)

for i in idxarr:
      if train[i].dtype == 'object':
        train[i] = le.fit_transform(train[i])
        test[i] = le.fit_transform(test[i])
# print(train.info())
# ------------------------------------------

# 피처임포턴스 그래프 보기 위해 데이터프레임형태의 x_, y_ 놔둠 / 훈련용 넘파이어레이형태의 x, y 생성-----------
x_ = train.drop(['ProdTaken','NumberOfChildrenVisiting','NumberOfPersonVisiting','OwnCar'], axis=1) # 피처임포턴스로 확인한 중요도 낮은 탑3 제거
# x_ = train.drop(['ProdTaken','NumberOfChildrenVisiting','NumberOfPersonVisiting','OwnCar', 'MonthlyIncome'], axis=1)
# x_ = train.drop(['ProdTaken'], axis=1)
y_ = train['ProdTaken']
x = np.array(x_)
y = np.array(y_)
# y = y.reshape(-1, 1) # y값 reshape 해야되서 x도 넘파이로 바꿔 훈련하는 것

test = test.drop(['NumberOfChildrenVisiting','NumberOfPersonVisiting','OwnCar'], axis=1) # 피처임포턴스로 확인한 중요도 낮은 탑3 제거
# test = test.drop(['NumberOfChildrenVisiting','NumberOfPersonVisiting','OwnCar', 'MonthlyIncome'], axis=1)
test = np.array(test)
# print(x.shape, y.shape)
#-----------------------------------------------------------------------------------------------------------

parameters_xgb = {
            'n_estimators':[400],
            'learning_rate':[0.3],
            'max_depth':[8],
            'gamma':[2],
            'min_child_weight':[5],
            'subsample':[0.7],
            'reg_alpha':[1],
            'reg_lambda':[0.001],
              } 

parameters_rnf = {
    'n_estimators':[300],
    'max_depth':[11],
    'min_samples_leaf':[11],
    'min_samples_split':[2],
    'n_jobs':[-1]
}

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=999, shuffle=True)
# print(np.unique(y_train, return_counts=True))

# 2. 모델
xgb = XGBClassifier(tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=0)
rnf = RandomForestClassifier(random_state=777) # 0.8951406649616368 / 1234     //  0.9028132992327366 // 777


# 3. 훈련
# model = xgb
model = rnf
# model = RandomizedSearchCV(xgb, parameters_xgb, cv=6, n_jobs=-1, verbose=2)
# model = GridSearchCV(rnf,  parameters_rnf, cv=5, n_jobs=-1, verbose=2)
# model = make_pipeline(MinMaxScaler(), HRS)
# model = make_pipeline(MinMaxScaler(), GridSearchCV(rnf, parameters_rnf, cv=5, n_jobs=-1, verbose=2))
# model = make_pipeline(MinMaxScaler(), xgb)
# model = make_pipeline(MinMaxScaler(), rnf)

import joblib
joblib.dump(model,'D:\study_data\_data\dacon_travel\_dat/m360_travel6.dat')
# model = joblib.load('D:\study_home\_data\dacon_travel\_dat/m360_travel.dat')


# 2. 모델
start = time.time()
model.fit(x_train, y_train)
end = time.time()

# 4. 평가, 예측
results = model.score(x_test, y_test)
print('스코어: ', results)
print('걸린 시간: ', end-start)

# 5. 제출 준비
y_submit = model.predict(test)

submission = pd.read_csv(filepath+'submission.csv', index_col=0)
submission['ProdTaken'] = y_submit
submission.to_csv(filepath + 'submission.csv', index = True)

# print(HRS.best_params_)



'''
 #   Column                    Non-Null Count  Dtype
---  ------                    --------------  -----
 0   Age                       1955 non-null   float64
 1   TypeofContact             1955 non-null   int32
 2   CityTier                  1955 non-null   int64
 3   DurationOfPitch           1955 non-null   float64
 4   Occupation                1955 non-null   int32
 5   Gender                    1955 non-null   int32
#  6   NumberOfPersonVisiting    1955 non-null   int64
 7   NumberOfFollowups         1955 non-null   float64
 8   ProductPitched            1955 non-null   int32
 9   PreferredPropertyStar     1955 non-null   float64
 10  MaritalStatus             1955 non-null   int32
 11  NumberOfTrips             1955 non-null   float64
 12  Passport                  1955 non-null   int64
 13  PitchSatisfactionScore    1955 non-null   int64
#  14  OwnCar                    1955 non-null   int64
#  15  NumberOfChildrenVisiting  1955 non-null   float64
 16  Designation               1955 non-null   int32
#  17  MonthlyIncome             1955 non-null   float64
 18  ProdTaken                 1955 non-null   int64
 '''
