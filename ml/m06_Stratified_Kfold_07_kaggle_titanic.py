from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.utils import all_estimators

# pandas의 y라벨의 종류 확인 train_set.columns.values
# numpy에서는 np.unique(y, return_counts=True)

# 1. 데이터
path = 'D:\study_data\_data\kaggle_titanic/'
train_set = pd.read_csv(path+'train.csv')
test_set = pd.read_csv(path+'test.csv')

print(train_set.describe())
print(train_set.info())
print(train_set.isnull())
print(train_set.isnull().sum())
print(train_set.shape) # (10886, 12)
print(train_set.columns.values) # ['PassengerId' 'Survived' 'Pclass' 'Name' 'Sex' 'Age' 'SibSp' 'Parch' 'Ticket' 'Fare' 'Cabin' 'Embarked']

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
y = np.array(y).reshape(-1, 1) # 벡터로 표시되어 있는 y데이터를 행렬로 전환

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8,shuffle=True, random_state=9)

from sklearn.preprocessing import MinMaxScaler
scaler= MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test) 
             
n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=99)

#2. 모델구성
from sklearn.svm import SVC
model = SVC()
    
# 3. 4. 컴파일, 훈련, 평가, 예측
# model.fit(x_train, y_train)
score = cross_val_score(model, x_train, y_train, cv=kfold) # 어차피 여기서 kfold에 대해서 straitify 분류 해준다 / 만약 다른 스코어를 쓰면 따로 stratify해주면 좋음
y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)
acc = accuracy_score(y_test, y_predict)

print('acc: ', score, '\n cross_val_score: ', round(np.mean(score),4))
print(y_predict)
print('cross_val_predict acc: ', acc)

# acc:  [0.85314685 0.72727273 0.81690141 0.81690141 0.83098592]
#  cross_val_score:  0.809
# [1 0 0 1 0 0 0 0 0 1 0 0 0 0 1 0 0 1 1 0 0 0 0 0 0 1 0 0 0 1 0 0 1 1 0 0 1
#  0 0 1 1 0 1 0 0 0 0 0 0 0 1 0 0 1 0 0 1 1 0 1 1 1 1 0 1 1 1 0 0 0 0 0 0 1
#  1 0 1 0 1 0 0 0 0 0 0 0 0 1 1 0 0 1 0 0 1 0 0 0 1 0 0 0 0 1 0 0 1 1 0 0 0
#  0 0 1 1 0 1 0 0 1 0 0 1 0 1 0 0 0 1 0 0 1 0 0 0 1 1 0 1 1 0 0 0 0 0 0 0 0
#  1 1 0 0 1 1 1 0 0 1 1 0 0 0 1 0 0 0 1 1 0 0 0 0 1 1 0 1 0 1 1]
# cross_val_predict acc:  0.7039106145251397