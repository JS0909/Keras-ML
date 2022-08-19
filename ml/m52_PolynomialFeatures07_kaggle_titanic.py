import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import time
import warnings
warnings.filterwarnings('ignore') # warnig 출력 안함

from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, MinMaxScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline



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

print(x.shape, y.shape) # (891, 7) (891, 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)

# 2. 모델
model = make_pipeline(MinMaxScaler(), RandomForestClassifier())
model.fit(x_train, y_train)
print('그냥 스코어: ',model.score(x_test, y_test))

from sklearn.model_selection import cross_val_score
kfold = KFold(n_splits=5, shuffle=True, random_state=1234)
score = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')
print('CV: ', score)
print('CV n빵: ', np.mean(score))
print('-------------------------------------------------------------')
#============================ PolynomialFeatures 후 ================================
pf = PolynomialFeatures(degree=2, include_bias=False) # include_bias = False 하면 기본으로 생기는 1이 안나옴
xp = pf.fit_transform(x)
print(xp.shape) # (891, 35)

x_train, x_test, y_train, y_test = train_test_split(xp, y, test_size=0.2, random_state=1234)

# 2. 모델
model = make_pipeline(MinMaxScaler(), RandomForestClassifier())
model.fit(x_train, y_train)
print('폴리스코어: ', model.score(x_test, y_test))

score = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')
print('폴리 CV: ', score)
print('폴리 CV n빵: ', np.mean(score))


# 그냥 스코어:  0.8268156424581006
# CV:  [0.81118881 0.76923077 0.76760563 0.85211268 0.81690141]
# CV n빵:  0.803407859745888
# -------------------------------------------------------------
# 폴리스코어:  0.8435754189944135
# 폴리 CV:  [0.78321678 0.76923077 0.75352113 0.84507042 0.80985915]
# 폴리 CV n빵:  0.792179651334581