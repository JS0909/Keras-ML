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
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import BaggingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer, PowerTransformer



# 1. 데이터
filepath = 'D:\study_data\_data\dacon_travel/'
train = pd.read_csv(filepath+'train.csv', index_col=0)
test = pd.read_csv(filepath+'test.csv', index_col=0)

# print(train.head())
# print(train.info())
# print(train.isnull().sum())

# 라벨 잘못 표기한거 수정
train = train.replace({'Gender' : 'Fe Male'}, 'Female')
test = test.replace({'Gender' : 'Fe Male'}, 'Female')

#===================================================================
# 행 삭제하면 서브미션파일과 행값 달라짐
# 천원받는 행 삭제
# didx = train.loc[train['MonthlyIncome']<=1000].index
# train.drop(didx, inplace=True)
# test.drop(didx, inplace=True)
# print(len(test))
# 프리랜서 한명 삭제
# didx = train.loc[train['Occupation']=='Free Lancer'].index
# train.drop(didx, inplace=True)
# test.drop(didx, inplace=True)
# print(len(test))
#===================================================================
train = train.replace({'Occupation':'Free Lancer'}, 'Small Business')
test = test.replace({'Occupation':'Free Lancer'}, 'Small Business')

train = train.replace({'MonthlyIncome': 1000.0}, 22995.0)


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

scaler = MinMaxScaler()
train[['Age','DurationOfPitch','MonthlyIncome']] = scaler.fit_transform(train[['Age','DurationOfPitch','MonthlyIncome']])
test[['Age','DurationOfPitch','MonthlyIncome']] = scaler.transform(test[['Age','DurationOfPitch','MonthlyIncome']])



# object타입 라벨인코딩--------------------
le = LabelEncoder()
idxarr = train.columns
idxarr = np.array(idxarr)

for i in idxarr:
      if train[i].dtype == 'object':
        train[i] = le.fit_transform(train[i])
        test[i] = le.fit_transform(test[i])
print(train.info())
# ------------------------------------------
# 원핫인코더
# train = pd.get_dummies(train)
# test = pd.get_dummies(test)
#-------------------------------------------

# 피처임포턴스 그래프 보기 위해 데이터프레임형태의 x_, y_ 놔둠 / 훈련용 넘파이어레이형태의 x, y 생성-----------
# x_ = train.drop(['ProdTaken','MonthlyIncome','NumberOfPersonVisiting','OwnCar'], axis=1) # 피처임포턴스로 확인한 중요도 낮은 탑3 제거
x_ = train.drop(['ProdTaken','NumberOfChildrenVisiting','NumberOfPersonVisiting','OwnCar', 'MonthlyIncome', 'NumberOfTrips','NumberOfFollowups'], axis=1)
# x_ = train.drop(['ProdTaken'], axis=1)
y_ = train['ProdTaken']
# y = y.reshape(-1, 1) # y값 reshape 해야되서 x도 넘파이로 바꿔 훈련하는 것

# test = test.drop(['MonthlyIncome','NumberOfPersonVisiting','OwnCar'], axis=1) # 피처임포턴스로 확인한 중요도 낮은 탑3 제거
test = test.drop(['NumberOfChildrenVisiting','NumberOfPersonVisiting','OwnCar', 'MonthlyIncome', 'NumberOfTrips','NumberOfFollowups'], axis=1)
test = np.array(test)
# print(x.shape, y.shape)
#-----------------------------------------------------------------------------------------------------------


x = np.array(x_)
y = np.array(y_)


parameters_xgb={
    'n_estimators':[500,600,700,800,1000],
    # 'learning_rate':[0.1,0.2,0.3,0.5,1,0.01,0.001],
    # 'max_depth':[None,2,3,4,5,6,7,8,9,10],
    # 'gamma':[0,1,2,3,4,5,7,10,100],
    # 'min_child_weight':[0,0.1,0.001,0.5,1,5,10,100],
    # 'subsample':[0,0.1,0.2,0.3,0.5,0.7,1],
    # 'colsample_bytree':[0,0.1,0.2,0.3,0.5,0.7,1],
    # 'colsample_bylevel':[0,0.1,0.2,0.3,0.5,0.7,1],
    # 'colsample_bynode':[0,0.1,0.2,0.3,0.5,0.7,1],
    # 'reg_alpha':[0,0.1,0.01,0.001,1,2,10],
    # 'reg_lambda':[0,0.1,0.01,0.001,1,2,10]
    }

parameters_rnf = {
    'n_estimators':[400],
    'max_depth':[None],
    'min_samples_leaf':[1],
    'min_samples_split':[2,3,5,7,10],
    'n_jobs':[-1]
}



x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=999, shuffle=True)
# print(np.unique(y_train, return_counts=True))

# smote = SMOTE(random_state=1234)
# x_train, y_train = smote.fit_resample(x_train, y_train)
# print(pd.Series(y_train).value_counts())

# 2. 모델
xgb = XGBClassifier(tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=0)
rnf = RandomForestClassifier(random_state=1234)

lg = LGBMClassifier()
cat = CatBoostClassifier(random_seed=0,learning_rate=0.5, verbose=0, bagging_temperature=66, subsample=0.5)

# 3. 훈련
# model = xgb
model = rnf
# model = cat
# model = VotingClassifier(estimators=[('XG', xgb), ('LG', lg), ('CAT', cat), ('RF', rnf)],
#                          voting='soft', verbose=2)
# model = GridSearchCV(xgb,  parameters_xgb, cv=5, n_jobs=-1, verbose=2)
# model = GridSearchCV(rnf,  parameters_rnf, cv=5, n_jobs=-1, verbose=2)
# model = make_pipeline(MinMaxScaler(), GridSearchCV(rnf, parameters_rnf, cv=5, n_jobs=-1, verbose=2))
# model = make_pipeline(MinMaxScaler(), xgb)
# model = make_pipeline(MinMaxScaler(), rnf)

import joblib
joblib.dump(model,'D:\study_data\_data\dacon_travel\_dat/m360_travel6.dat')
# model = joblib.load('D:\study_home\_data\dacon_travel\_dat/m360_travel.dat')


# 2. 모델
model.fit(x_train, y_train)

# 4. 평가, 예측
results = model.score(x_test, y_test)
print(results)

# 5. 제출 준비
model.fit(x,y)
y_submit = model.predict(test)
# print(len(y_submit))


submission = pd.read_csv(filepath+'submission.csv', index_col=0)
submission['ProdTaken'] = y_submit
submission.to_csv(filepath + 'submission.csv', index = True)

# print(model.best_params_)
# print(np.argsort(rnf.feature_importances_))
# [ 9 29 18 20 25 28 19 27 17 21 24 11 12 26 10 13  8 23 16 14 22 15  1  4
#   3  7  5  6  2  0]

# 1379
# 스코어:  0.9002557544757033

# 1379
# 스코어:  0.9028132992327366

# results: 0.9053708439897699 베스트

# 0.907928388746803

# 0.8925831202046036

# 0.9028132992327366

# 0.9181585677749361 // rnf_randomstate= 1234

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
