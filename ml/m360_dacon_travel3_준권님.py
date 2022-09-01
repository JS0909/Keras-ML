from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score

import time
import pandas_profiling
import matplotlib
matplotlib.rcParams['font.family']='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus']=False
import pandas as pd
import numpy as np
import optuna
from optuna import Trial, visualization
from optuna.visualization import plot_parallel_coordinate
from optuna.samplers import TPESampler

#1. 데이터
path =  'D:\study_data\_data\dacon_travel/' # ".은 현재 폴더"
train_set = pd.read_csv(path + 'train.csv',
                        index_col=0)
# print(train_set.info())
test_set = pd.read_csv(path + 'test.csv', #예측에서 쓸거야!!
                       index_col=0)
# print(train_set.shape,test_set.shape) (1955, 19) (2933, 18)                       
# Data columns (total 19 columns):
#  #   Column                    Non-Null Count  Dtype
# ---  ------                    --------------  -----
#  0   Age                       1861 non-null   float64  MonthlyIncome
#  1   TypeofContact             1945 non-null   object # 빈도로 메꾸기 
#  2   CityTier                  1955 non-null   int64 
#  3   DurationOfPitch           1853 non-null   float64 앞뒤행으로 
#  4   Occupation                1955 non-null   object
#  5   Gender                    1955 non-null   object
#  6   NumberOfPersonVisiting    1955 non-null   int64
#  7   NumberOfFollowups         1942 non-null   float64  
#  8   ProductPitched            1955 non-null   object
#  9   PreferredPropertyStar     1945 non-null   float64
#  10  MaritalStatus             1955 non-null   object
#  11  NumberOfTrips             1898 non-null   float64
#  12  Passport                  1955 non-null   int64
#  13  PitchSatisfactionScore    1955 non-null   int64
#  14  OwnCar                    1955 non-null   int64
#  15  NumberOfChildrenVisiting  1928 non-null   float64
#  16  Designation               1955 non-null   object
#  17  MonthlyIncome             1855 non-null   float64
#  18  ProdTaken                 1955 non-null   int64
######### 결측치 채우기 (클래스별 괴리가 큰 컬럼으로 평균 채우기)

# train_set = train_set.drop([190,605,1339],inplace=True)
# test_set = test_set.drop(index3,index=True)

train_set['TypeofContact'].fillna('Self Enquiry', inplace=True)
test_set['TypeofContact'].fillna('Self Enquiry', inplace=True)
train_set['Age'].fillna(train_set.groupby('Designation')['Age'].transform('mean'), inplace=True)
test_set['Age'].fillna(test_set.groupby('Designation')['Age'].transform('mean'), inplace=True)
train_set['Age']=np.round(train_set['Age'],0).astype(int)
test_set['Age']=np.round(test_set['Age'],0).astype(int)

# print(train_set.isnull().sum()) #(1955, 19)
# print(train_set[train_set['MonthlyIncome'].notnull()].groupby(['Designation'])['MonthlyIncome'].mean())

train_set['MonthlyIncome'].fillna(train_set.groupby('Designation')['MonthlyIncome'].transform('mean'), inplace=True)
test_set['MonthlyIncome'].fillna(test_set.groupby('Designation')['MonthlyIncome'].transform('mean'), inplace=True)
# print(train_set.describe) #(1955, 19)
# print(train_set[train_set['DurationOfPitch'].notnull()].groupby(['NumberOfChildrenVisiting'])['DurationOfPitch'].mean())
train_set['DurationOfPitch'].fillna(train_set['DurationOfPitch'].median(), inplace=True)
test_set['DurationOfPitch'].fillna(test_set['DurationOfPitch'].median(), inplace=True)

# print(train_set[train_set['NumberOfFollowups'].notnull()].groupby(['NumberOfChildrenVisiting'])['NumberOfFollowups'].mean())
train_set['NumberOfFollowups'].fillna(train_set.groupby('NumberOfChildrenVisiting')['NumberOfFollowups'].transform('mean'), inplace=True)
test_set['NumberOfFollowups'].fillna(test_set.groupby('NumberOfChildrenVisiting')['NumberOfFollowups'].transform('mean'), inplace=True)
# print(train_set[train_set['PreferredPropertyStar'].notnull()].groupby(['Occupation'])['PreferredPropertyStar'].mean())
train_set['PreferredPropertyStar'].fillna(train_set.groupby('Occupation')['PreferredPropertyStar'].transform('mean'), inplace=True)
test_set['PreferredPropertyStar'].fillna(test_set.groupby('Occupation')['PreferredPropertyStar'].transform('mean'), inplace=True)

# train_set['AgeBound'] = pd.cut(train_set['Age'], 5)
# 임의로 5개 그룹을 지정
# print(train_set['IncomeBand'])
# 나이: [(17.957, 26.6] < (26.6, 35.2] < (35.2, 43.8] < (43.8, 52.4] < (52.4, 61.0]]

combine = [train_set,test_set]
for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 29, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 29) & (dataset['Age'] <= 39), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 39) & (dataset['Age'] <= 49), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 49) & (dataset['Age'] <= 59), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 59, 'Age'] = 4
# train_set = train_set.drop(['AgeBand'], axis=1)

train_set['NumberOfTrips'].fillna(train_set.groupby('DurationOfPitch')['NumberOfTrips'].transform('mean'), inplace=True)
test_set['NumberOfTrips'].fillna(test_set.groupby('DurationOfPitch')['NumberOfTrips'].transform('mean'), inplace=True)
train_set['NumberOfChildrenVisiting'].fillna(train_set.groupby('MaritalStatus')['NumberOfChildrenVisiting'].transform('mean'), inplace=True)
test_set['NumberOfChildrenVisiting'].fillna(test_set.groupby('MaritalStatus')['NumberOfChildrenVisiting'].transform('mean'), inplace=True)

train_set.loc[ train_set['Gender'] =='Fe Male' , 'Gender'] = 'Female'
test_set.loc[ test_set['Gender'] =='Fe Male' , 'Gender'] = 'Female'
cols = ['TypeofContact','Occupation','Gender','ProductPitched','MaritalStatus','Designation']

#===================================================================
# 천원받는 행 삭제
# didx = train_set.loc[train_set['MonthlyIncome']<=1000].index
# train_set.drop(didx, inplace=True)

# 프리랜서 한명 삭제
# didx = train_set.loc[train_set['Occupation']=='Free Lancer']
# train_set.drop(didx, inplace=True)

# 프리랜서 -> 월급쟁이로
train_set.loc[train_set['Occupation']=='Free Lancer', 'Occupation'] = 'Salaried'

# 커플 여부 (결혼 / 싱글+이혼+미혼)
# train_set.loc[train_set['MaritalStatus']=='Married', 'MaritalStatus'] = 'couple'
# train_set.loc[train_set['MaritalStatus']=='Unmarried', 'MaritalStatus'] = 'single'

# train_set.loc[train_set['MaritalStatus']=='Divorced', 'MaritalStatus'] = 'single'
# train_set.loc[train_set['MaritalStatus']=='Single', 'MaritalStatus'] = 'single'

# 이렇게하면 비율 비슷해지긴 함
# print(train_set['MaritalStatus'].value_counts())
#===================================================================

from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm_notebook

for col in iter(tqdm_notebook(cols)): # 프로그래스바 포함시킨거
    le = LabelEncoder()
    train_set[col]=le.fit_transform(train_set[col])
    test_set[col]=le.fit_transform(test_set[col])
# print(train_set['TypeofContact'])

# html 파일로 전체적인 데이터 모양 시각화
# profile = train_set.profile_report()
# profile.to_file(output_file = 'train_set.html')

def outliers(data_out):
    quartile_1, q2 , quartile_3 = np.percentile(data_out,
                                               [25,50,75]) # percentile 백분위
    # print("1사분위 : ",quartile_1) # 25% 위치인수를 기점으로 사이에 값을 구함
    # print("q2 : ",q2) # 50% median과 동일 
    # print("3사분위 : ",quartile_3) # 75% 위치인수를 기점으로 사이에 값을 구함
    iqr =quartile_3-quartile_1  # 75% -25%
    # print("iqr :" ,iqr)
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((data_out>upper_bound)|
                    (data_out<lower_bound))


# Age_out_index= outliers(train_set['Age'])[0]
TypeofContact_out_index= outliers(train_set['TypeofContact'])[0]
CityTier_out_index= outliers(train_set['CityTier'])[0]
DurationOfPitch_out_index= outliers(train_set['DurationOfPitch'])[0]
Gender_out_index= outliers(train_set['Gender'])[0]
NumberOfPersonVisiting_out_index= outliers(train_set['NumberOfPersonVisiting'])[0]
NumberOfFollowups_out_index= outliers(train_set['NumberOfFollowups'])[0]
ProductPitched_index= outliers(train_set['ProductPitched'])[0]
PreferredPropertyStar_out_index= outliers(train_set['PreferredPropertyStar'])[0]
MaritalStatus_out_index= outliers(train_set['MaritalStatus'])[0]
NumberOfTrips_out_index= outliers(train_set['NumberOfTrips'])[0]
Passport_out_index= outliers(train_set['Passport'])[0]
PitchSatisfactionScore_out_index= outliers(train_set['PitchSatisfactionScore'])[0]
OwnCar_out_index= outliers(train_set['OwnCar'])[0]
NumberOfChildrenVisiting_out_index= outliers(train_set['NumberOfChildrenVisiting'])[0]
Designation_out_index= outliers(train_set['Designation'])[0]
MonthlyIncome_out_index= outliers(train_set['MonthlyIncome'])[0]

# print(DurationOfPitch_out_index)

lead_outlier_index = np.concatenate((#Age_out_index,                            # acc : 0.8650306748466258
                                    #  TypeofContact_out_index,                 # acc : 0.8920454545454546
                                    #  CityTier_out_index,                      # acc : 0.8920454545454546
                                     DurationOfPitch_out_index,               # acc : 0.9156976744186046
                                    #  Gender_out_index,                        # acc : 0.8920454545454546
                                    #  NumberOfPersonVisiting_out_index,        # acc : 0.8835227272727273
                                    #  NumberOfFollowups_out_index,             # acc : 0.8942598187311178
                                    #  ProductPitched_index,                    # acc : 0.8920454545454546
                                    #  PreferredPropertyStar_out_index,         # acc : 0.8920454545454546
                                    #  MaritalStatus_out_index,                 # acc : 0.8920454545454546
                                    #  NumberOfTrips_out_index,                 # acc : 0.8670520231213873
                                    #  Passport_out_index,                      # acc : 0.8920454545454546
                                    #  PitchSatisfactionScore_out_index,        # acc : 0.8920454545454546
                                    #  OwnCar_out_index,                        # acc : 0.8920454545454546
                                    #  NumberOfChildrenVisiting_out_index,      # acc : 0.8920454545454546
                                    #  Designation_out_index,                   # acc : 0.8869047619047619
                                    #  MonthlyIncome_out_index                  # acc : 0.8932926829268293
                                     ),axis=None)
# print(len(lead_outlier_index)) #577
# print(lead_outlier_index)

lead_not_outlier_index = []
for i in train_set.index:
    if i not in lead_outlier_index :
        lead_not_outlier_index.append(i)
train_set_clean = train_set.loc[lead_not_outlier_index]      
train_set_clean = train_set_clean.reset_index(drop=True)
# print(train_set_clean)
x = train_set_clean.drop(['ProdTaken','NumberOfChildrenVisiting','NumberOfPersonVisiting','OwnCar', 'NumberOfFollowups', 'MonthlyIncome', 'Designation', 'NumberOfTrips'], axis=1)
# x = train_set_clean.drop(['ProdTaken'], axis=1)
test_set = test_set.drop(['NumberOfChildrenVisiting','NumberOfPersonVisiting','OwnCar', 'NumberOfFollowups', 'MonthlyIncome', 'Designation', 'NumberOfTrips'], axis=1)
y = train_set_clean['ProdTaken']
print(x.shape)


from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.preprocessing import StandardScaler,MinMaxScaler 
from xgboost import XGBClassifier,XGBRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from imblearn.over_sampling import SMOTE

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.91,shuffle=True,random_state=1234,stratify=y)
# smote = SMOTE(random_state=123)
# x_train,y_train = smote.fit_resample(x_train,y_train)

from xgboost import XGBClassifier,XGBRegressor
from sklearn.metrics import r2_score,accuracy_score
from catboost import CatBoostRegressor,CatBoostClassifier
from bayes_opt import BayesianOptimization
# 2. 모델
# ''' 그리드 서치
n_splits = 6

kfold = StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=123)

cat_paramets = {"learning_rate" : [0.20909079092170735],
                'depth' : [8],
                'od_pval' : [0.236844398775451],
                'model_size_reg': [0.30614059763442997],
                'l2_leaf_reg' :[5.535171839105427]}
cat = CatBoostClassifier(random_state=123,verbose=False,n_estimators=500)
model = GridSearchCV(cat,cat_paramets,cv=kfold,n_jobs=-1)
# '''
'''
# optuna
def objectiveCAT(trial: Trial, x_train, y_train, x_test):
    param = {
        'n_estimators' : trial.suggest_int('n_estimators', 100, 1000),
        'depth' : trial.suggest_int('depth', 5, 16),
        'fold_permutation_block' : trial.suggest_int('fold_permutation_block', 1, 256),
        'learning_rate' : trial.suggest_float('learning_rate', 0.001, 1),
        'od_pval' : trial.suggest_float('od_pval', 0, 1),
        'l2_leaf_reg' : trial.suggest_float('l2_leaf_reg', 0, 8),
        'random_state' : 1127
    }
    # 학습 모델 생성
    model = CatBoostClassifier(**param)
    CAT_model = model.fit(x_train, y_train, verbose=0) # 학습 진행
    
    # 모델 성능 확인
    score = accuracy_score(CAT_model.predict(x_test), y_test)
    
    return score

# MAE가 최소가 되는 방향으로 학습을 진행
# TPESampler : Sampler using TPE (Tree-structured Parzen Estimator) algorithm.
study = optuna.create_study(direction='maximize', sampler=TPESampler())

# n_trials 지정해주지 않으면, 무한 반복
study.optimize(lambda trial : objectiveCAT(trial, x, y, x_test), n_trials = 1)

print('Best trial : score {}, \nparams {}'.format(study.best_trial.value, study.best_trial.params))
'''

# model = CatBoostClassifier(n_estimators= 951, depth= 5, fold_permutation_block= 54, 
#         learning_rate= 0.8939968681408258, od_pval= 0.15607692947279805, l2_leaf_reg= 0.216759718976407, verbose=0)

# model = CatBoostClassifier(n_estimators= 748, depth= 12, fold_permutation_block= 235, 
#         learning_rate= 0.9400406267671246, od_pval= 0.048469909084309926, l2_leaf_reg= 0.927785247837019, verbose=0)

# model = CatBoostClassifier(n_estimators= 561, depth= 8, fold_permutation_block= 134, 
#         learning_rate= 0.5610934137500165, od_pval= 0.3492486557791592, l2_leaf_reg= 1.3928694532637893, verbose=0)

start_time = time.time()
model.fit(x_train,y_train)   
end_time = time.time()-start_time 
y_predict = model.predict(x_test)
results = accuracy_score(y_test,y_predict)
# print('최적의 매개변수 : ',model.best_params_)
# print('최상의 점수 : ', model.best_score_)
print('acc :',results)
# print('걸린 시간 :',end_time)


model.fit(x,y)
y_summit = model.predict(test_set)
y_summit = np.round(y_summit,0)
submission = pd.read_csv(path + 'sample_submission.csv',#예측에서 쓸거야!!
                      )
submission['ProdTaken'] = y_summit

submission.to_csv(path+'submissionJun.csv',index=False)


##########
# 최상의 점수 :  0.8930338463986
# acc : 0.9418604651162791
# 걸린 시간 : 11.291642665863037

############ RandomState = 100
# 최상의 점수 :  0.8813139873889755
# acc : 0.921875
# 걸린 시간 : 7.259145259857178


# 최상의 점수 :  0.9010738575349003
# acc : 0.9244186046511628
# 걸린 시간 : 4.313105344772339

# 최상의 점수 :  0.9068309270970052
# acc : 0.936046511627907

# 지금 베스트(나이만 범주화)
# 최상의 점수 :  0.9045300878972279
# acc : 0.9534883720930233

# 프리랜서한명만 삭제
# 최상의 점수 :  0.9159567275185937
# acc : 0.936046511627907

# 예전 베스트
# 최상의 점수 :  0.9079485343833272
# acc : 0.9476744186046512

# 두번째 베스트
# 최상의 점수 :  0.9062582030783917
# acc : 0.9593023255813954

# 결혼여부 : 이혼 / 싱글+미혼 / 기혼
# 최상의 점수 :  0.9062621803285208
# acc : 0.9534883720930233