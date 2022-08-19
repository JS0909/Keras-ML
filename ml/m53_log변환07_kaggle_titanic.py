from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, RobustScaler, StandardScaler,\
    QuantileTransformer, PowerTransformer # = 이상치에 자유로운 편
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

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

x_train, x_test, y_train, y_test = train_test_split(x, y.values.ravel(), test_size=0.2, random_state=1234)

# scl = StandardScaler()
# x_train = scl.fit_transform(x_train)
# x_test = scl.transform(x_test)

# 2. 모델
# model = LogisticRegression()
model = RandomForestClassifier(random_state=1234)

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
y_pred = model.predict(x_test)
result = r2_score(y_test, y_pred)
print('그냥 결과: ', round(result,4))

# 

#=================== 로그 변환 ======================
print(x.columns)

x.plot.box()
plt.title('covtype')
plt.xlabel('Feature')
plt.ylabel('데이터값')
plt.show()

# x['magnesium'] = np.log1p(x['magnesium']) #  0.8946

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)

# scl = StandardScaler()
# x_train = scl.fit_transform(x_train)
# x_test = scl.transform(x_test)

# 2. 모델
# model = LogisticRegression()
model = RandomForestClassifier(random_state=1234)

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
y_pred = model.predict(x_test)
result = accuracy_score(y_test, y_pred)
print('로그 변환 결과: ', round(result,4))