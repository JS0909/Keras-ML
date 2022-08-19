import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import time
import warnings
warnings.filterwarnings('ignore') # warnig 출력 안함

from sklearn.model_selection import train_test_split, KFold

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import r2_score, accuracy_score

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, RobustScaler, StandardScaler,\
    QuantileTransformer, PowerTransformer



# 1. 데이터
path = 'D:\study_data\_data\kaggle_titanic/'
train_set = pd.read_csv(path+'train.csv')
test_set = pd.read_csv(path+'test.csv')

train_set = train_set.drop(columns='Cabin', axis=1)
train_set['Age'].fillna(train_set['Age'].mean(), inplace=True)
print(train_set['Embarked'].mode())
train_set['Embarked'].fillna(train_set['Embarked'].mode()[0], inplace=True)
train_set.replace({'Sex':{'male':0,'female':1}, 'Embarked':{'S':0,'C':1,'Q':2}}, inplace=True)

# train_set 불러올 때와 마찬가지로 전처리시켜야 model.predict에 넣어서 y값 구하기가 가능함-----------
print(test_set.isnull().sum())
test_set = test_set.drop(columns='Cabin', axis=1)
test_set['Age'].fillna(test_set['Age'].mean(), inplace=True)
test_set['Fare'].fillna(test_set['Fare'].mean(), inplace=True)
print(test_set['Embarked'].mode())
test_set['Embarked'].fillna(test_set['Embarked'].mode()[0], inplace=True)
test_set.replace({'Sex':{'male':0,'female':1}, 'Embarked':{'S':0,'C':1,'Q':2}}, inplace=True)
test_set = test_set.drop(columns = ['PassengerId','Name','Ticket'],axis=1)
#---------------------------------------------------------------------------------------------------

y = train_set['Survived']
x = train_set.drop(columns = ['PassengerId','Name','Ticket','Survived'],axis=1)
print(x)

x = np.array(x)
y = np.array(y).reshape(-1, 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)

model = RandomForestClassifier()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
result = r2_score(y_test, y_pred)
print('no scaler: ', round(result,4))



sclist = [StandardScaler(),MinMaxScaler(),MaxAbsScaler(),RobustScaler(),QuantileTransformer(),
          PowerTransformer(method='yeo-johnson'), # 디폴트 
          PowerTransformer(method='box-cox')
            ]
               
for scl in sclist:

    if str(scl) == str(PowerTransformer(method='box-cox')):
        try:
            scl = PowerTransformer(method='box-cox')
            x_train = scl.fit_transform(x_train)
        except:
            print('box-cox 안됨')
            break
          
    x_train = scl.fit_transform(x_train)
    x_test = scl.transform(x_test)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    result = accuracy_score(y_test, y_pred)
    print(str(scl).strip('()')+'결과: ', round(result,4))
    

# no scaler:  0.2493
# StandardScaler결과:  0.8212
# MinMaxScaler결과:  0.8156
# MaxAbsScaler결과:  0.8268
# RobustScaler결과:  0.8324
# QuantileTransformer결과:  0.8156
# PowerTransformer결과:  0.8324
# box-cox 안됨