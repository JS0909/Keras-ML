from cProfile import label
import os
from re import I, X
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 준비
path='D:\study_data\_data\dacon_travel/'
train_set=pd.read_csv(path+'train.csv')
submission=pd.read_csv(path+'sample_submission.csv',index_col=0)
test_set=pd.read_csv(path+'test.csv') #예측할때 사용할거에요!!

# 고객의 제품 인지 방법 (회사의 홍보 or 스스로 검색) mapping
for these in [train_set, test_set]:
    these['TypeofContact'] = these['TypeofContact'].map({'Unknown': 0, 'Company Invited': 2, 'Self Enquiry': 1})

# 성별 mapping
for these in [train_set, test_set]:
    these['Gender'] = these['Gender'].map({'Male': 0, 'Female': 1, 'Fe Male': 1})

# 직업 mapping
for these in [train_set, test_set]:
    these['Occupation'] = these['Occupation'].map({'Salaried': 0, 'Small Business': 1, 'Large Business': 2, 'Free Lancer':3})

# 영업 사원이 제시한 상품 mapping
for these in [train_set, test_set]:
    these['ProductPitched'] = these['ProductPitched'].map({'Super Deluxe': 0, 'King': 1, 'Deluxe': 2, 'Standard':3, 'Basic': 4})

# 결혼 여부 mapping
for these in [train_set, test_set]:
    these['MaritalStatus'] = these['MaritalStatus'].map({'Divorced': 0, 'Married': 1, 'Unmarried': 2, 'Single':3})

# 직급 mapping
for these in [train_set, test_set]:
    these['Designation'] = these['Designation'].map({'AVP': 0, 'VP': 1, 'Manager': 2, 'Senior Manager':3, 'Executive': 4})


# 결측치 처리
ls = ['TypeofContact', 'DurationOfPitch', ]
for these in [train_set, test_set]:
    for col in ls:
        these[col].fillna(0, inplace=True)

mean_cols = ['Age','NumberOfFollowups','PreferredPropertyStar', 'MonthlyIncome',
            'NumberOfTrips','NumberOfChildrenVisiting']
for these in [train_set, test_set]:
    for col in mean_cols:
        these[col] = these[col].fillna(train_set[col].mean())

print(train_set.info())

# 스케일링
scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = RobustScaler()
# scaler = MaxAbsScaler()
train_set[['Age', 'DurationOfPitch', 'MonthlyIncome']] = scaler.fit_transform(train_set[['Age', 'DurationOfPitch', 'MonthlyIncome']])
test_set[['Age', 'DurationOfPitch', 'MonthlyIncome']] = scaler.transform(test_set[['Age', 'DurationOfPitch', 'MonthlyIncome']])

# train을 x, y로 나누고 불필요한 컬럼 제거
train_set.drop(columns=['id'], inplace=True)
test_set.drop(columns=['id'], inplace=True)
x = train_set.drop(columns=['ProdTaken'])
y = train_set[['ProdTaken']]


from catboost import CatBoostClassifier


model = CatBoostClassifier()
model.fit(x, y.values.ravel())
score=model.score(x,y)
# print('score:',score) score: 1.0

# 데이터 submit
y_summit = model.predict(test_set)
submission['ProdTaken'] = y_summit
submission.to_csv(path + 'submissionJ.csv', index = True)

