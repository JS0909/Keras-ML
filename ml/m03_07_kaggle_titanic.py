# [실습]
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from tensorflow.python.keras.callbacks import EarlyStopping

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


#2. 모델구성
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron, LogisticRegression # !논리회귀(분류임)!
from sklearn.neighbors import KNeighborsClassifier # 최근접 이웃
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier # 결정트리를 여러개 랜덤으로 뽑아서 앙상블해서 봄

model = LinearSVC()
model.fit(x_train, y_train)
result = model.score(x_test, y_test)
print('LinearSVC acc 결과: ', result)
# y_predict = model.predict(x_test)
# print('ypred: ', y_predict, '\n')

model = SVC()
model.fit(x_train, y_train)
result = model.score(x_test, y_test)
print('SVC acc 결과: ', result)
# y_predict = model.predict(x_test)
# print('ypred: ', y_predict, '\n')

model = Perceptron()
model.fit(x_train, y_train)
result = model.score(x_test, y_test)
print('Perceptron acc 결과: ', result)
# y_predict = model.predict(x_test)
# print('ypred: ', y_predict, '\n')

model = LogisticRegression()
model.fit(x_train, y_train)
result = model.score(x_test, y_test)
print('LogisticRegression acc 결과: ', result)
# y_predict = model.predict(x_test)
# print('ypred: ', y_predict, '\n')

model = KNeighborsClassifier()
model.fit(x_train, y_train)
result = model.score(x_test, y_test)
print('KNeighborsClassifier acc 결과: ', result)
# y_predict = model.predict(x_test)
# print('ypred: ', y_predict, '\n')

model = DecisionTreeClassifier()
model.fit(x_train, y_train)
result = model.score(x_test, y_test)
print('DecisionTreeClassifier acc 결과: ', result)
# y_predict = model.predict(x_test)
# print('ypred: ', y_predict, '\n')

model = RandomForestClassifier()
model.fit(x_train, y_train)
result = model.score(x_test, y_test)
print('RandomForestClassifier acc 결과: ', result)
# y_predict = model.predict(x_test)
# print('ypred: ', y_predict, '\n')

# 5. 제출 준비-------------------------------------------------------
# submission = pd.read_csv(path + 'gender_submission.csv', index_col=0)

# y_submit = model.predict(test_set)
# y_submit = np.round(y_submit)
# y_submit = y_submit.astype(int)

# submission['Survived'] = y_submit
# submission.to_csv(path + 'gender_submission.csv', index=True)
#--------------------------------------------------------------------


# SVC acc 결과:  0.6312849162011173
# Perceptron acc 결과:  0.6815642458100558
# LogisticRegression acc 결과:  0.770949720670391
# KNeighborsClassifier acc 결과:  0.6759776536312849
# DecisionTreeClassifier acc 결과:  0.7653631284916201
# RandomForestClassifier acc 결과:  0.776536312849162