import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
import optuna
from optuna import Trial
from optuna.samplers import TPESampler
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import joblib as jl


#1. data
'''
id : 샘플 아이디
Age : 나이
TypeofContact : 고객의 제품 인지 방법 (회사의 홍보 or 스스로 검색)
CityTier : 주거 중인 도시의 등급. (인구, 시설, 생활 수준 기준) (1등급 > 2등급 > 3등급) 
DurationOfPitch : 영업 사원이 고객에게 제공하는 프레젠테이션 기간
Occupation : 직업
Gender : 성별
NumberOfPersonVisiting : 고객과 함께 여행을 계획 중인 총 인원
NumberOfFollowups : 영업 사원의 프레젠테이션 후 이루어진 후속 조치 수
ProductPitched : 영업 사원이 제시한 상품
PreferredPropertyStar : 선호 호텔 숙박업소 등급
MaritalStatus : 결혼여부
NumberOfTrips : 평균 연간 여행 횟수
Passport : 여권 보유 여부 (0: 없음, 1: 있음)
PitchSatisfactionScore : 영업 사원의 프레젠테이션 만족도
OwnCar : 자동차 보유 여부 (0: 없음, 1: 있음)
NumberOfChildrenVisiting : 함께 여행을 계획 중인 5세 미만의 어린이 수
Designation : (직업의) 직급
MonthlyIncome : 월 급여
ProdTaken : 여행 패키지 신청 여부 (0: 신청 안 함, 1: 신청함)
'''

path = 'D:\study_home\_data\_travel/'
train_set = pd.read_csv(path + 'train.csv', index_col=0)
test_set = pd.read_csv(path + 'test.csv', index_col=0)

train_set.loc[ train_set['Gender'] == 'Fe Male', 'Gender'] = 'Female'
test_set.loc[ test_set['Gender'] == 'Fe Male', 'Gender'] = 'Female'

train_set.loc[ train_set['Occupation'] == 'Free Lancer', 'Occupation'] = 'Salaried'
test_set.loc[ test_set['Occupation'] == 'Free Lancer', 'Occupation'] = 'Salaried'

train_set['TypeofContact'].fillna('Self Enquiry', inplace=True)
test_set['TypeofContact'].fillna('Self Enquiry', inplace=True)

print(train_set.groupby('Designation')['Age'].median())
train_set['Age'].fillna(train_set.groupby('Designation')['Age'].transform('mean'), inplace=True)
test_set['Age'].fillna(test_set.groupby('Designation')['Age'].transform('mean'), inplace=True)

train_set['Age'] = np.round(train_set['Age'], 0).astype(int)
test_set['Age'] = np.round(test_set['Age'], 0).astype(int)
datasets = [train_set,test_set]
for dataset in datasets:    
    dataset.loc[ dataset['Age'] <= 20, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 20) & (dataset['Age'] <= 29), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 29) & (dataset['Age'] <= 39), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 39) & (dataset['Age'] <= 49), 'Age'] = 3
    dataset.loc[(dataset['Age'] > 49) & (dataset['Age'] <= 59), 'Age'] = 4
    dataset.loc[ dataset['Age'] > 59, 'Age'] = 5


# train_set['MonthlyIncome'].fillna(train_set.groupby('Designation')['MonthlyIncome'].transform('mean'), inplace=True)
# test_set['MonthlyIncome'].fillna(test_set.groupby('Designation')['MonthlyIncome'].transform('mean'), inplace=True)

train_set['DurationOfPitch']=train_set['DurationOfPitch'].fillna(0)
test_set['DurationOfPitch']=test_set['DurationOfPitch'].fillna(0)

train_set['PreferredPropertyStar'].fillna(train_set.groupby('Occupation')['PreferredPropertyStar'].transform('mean'), inplace=True)
test_set['PreferredPropertyStar'].fillna(test_set.groupby('Occupation')['PreferredPropertyStar'].transform('mean'), inplace=True)

train_set['NumberOfTrips'].fillna(train_set.groupby('DurationOfPitch')['NumberOfTrips'].transform('mean'), inplace=True)
test_set['NumberOfTrips'].fillna(test_set.groupby('DurationOfPitch')['NumberOfTrips'].transform('mean'), inplace=True)

le = LabelEncoder()
cols = np.array(train_set.columns)
for c in cols:
      if train_set[c].dtype == 'object':
        train_set[c] = le.fit_transform(train_set[c])
        test_set[c] = le.fit_transform(test_set[c])

# checking outliers
def outliers(data_out):
    quartile_1, q2 , quartile_3 = np.percentile(data_out, [25,50,75])
    iqr =quartile_3-quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((data_out>upper_bound)|
                    (data_out<lower_bound))
    
outliers_list=[]
def outliers_printer(dataset):
    plt.figure(figsize=(10,8))
    for i in range(dataset.shape[1]):
        col = dataset[:, i]
        outliers_loc = outliers(col)
        print(i, '열의 이상치의 위치: ', outliers_loc, '\n')
        plt.subplot(math.ceil(dataset.shape[1]/2),2,i+1)
        plt.boxplot(col)
        plt.title(i)
        outliers_list.append([outliers_loc[0]])
        
    plt.show()
                     
outliers_printer(train_set.values)
plt.boxplot(train_set['DurationOfPitch'])                           
plt.show()

x = train_set.drop(['ProdTaken','NumberOfPersonVisiting','NumberOfChildrenVisiting', 'OwnCar', 'MonthlyIncome', 'NumberOfFollowups'], axis=1)
test_set = test_set.drop(['NumberOfPersonVisiting','NumberOfChildrenVisiting','OwnCar', 'MonthlyIncome', 'NumberOfFollowups'], axis=1)
y = train_set['ProdTaken']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, shuffle=True, random_state=1234, stratify=y)


# 2. model

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
    # model
    model = CatBoostClassifier(**param)
    CAT_model = model.fit(x_train, y_train, verbose=0)
    
    score = accuracy_score(CAT_model.predict(x_test), y_test)
    
    return score

# TPESampler : Sampler using TPE (Tree-structured Parzen Estimator) algorithm.
study = optuna.create_study(direction='maximize', sampler=TPESampler())

study.optimize(lambda trial : objectiveCAT(trial, x, y, x_test), n_trials = 1)

print('Best trial : score {}, \nparams {}'.format(study.best_trial.value, study.best_trial.params))


# Best trial : score 1.0,
# params {'n_estimators': 1304,'depth': 8, 'fold_permutation_block': 142, 'learning_rate': 0.21616891196578603, 
#         'od_pval': 0.12673190617341812, 'l2_leaf_reg': 0.33021257848638497}

kfold = StratifiedKFold(n_splits=6, shuffle=True, random_state=123)
cat_parameters = {"learning_rate" : [0.01],
                'depth' : [8],
                'od_pval' : [0.12673190617341812],
                'fold_permutation_block': [142],
                'l2_leaf_reg' :[0.33021257848638497]}
cat = CatBoostClassifier(n_estimators=1304, random_state=72, verbose=0)
model = GridSearchCV(cat, cat_parameters, cv=kfold)
model.fit(x_train,y_train)   


# 4. evaluate
y_predict = model.predict(x_test)
score = accuracy_score(y_test,y_predict)
print('acc :', score)


# 5. preparing submission
model.fit(x,y)
y_submit = model.predict(test_set)
y_submit = np.round(y_submit, 0)
submission = pd.read_csv(path + 'sample_submission.csv')
submission['ProdTaken'] = y_submit

submission.to_csv(path+'submission.csv',index=False)


# 6. saving model & weights
jl.dump(model, path + 'save.dat')


# acc : 0.9795918367346939
