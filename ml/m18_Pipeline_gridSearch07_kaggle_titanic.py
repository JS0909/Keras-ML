import numpy as np
import pandas as pd
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, StratifiedKFold,\
    HalvingRandomSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler


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

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=1234)

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=99)

parameters = [ # 명시한 모델명을 앞에 언더바 두개 해주고 붙여넣어줘야 파이프+그리드서치 적용됨
    {'RF__n_estimators':[100,200], 'RF__max_depth':[6,8,10,12], 'RF__n_jobs':[-1]},
    {'RF__max_depth':[6,8,10,12], 'RF__min_samples_leaf':[3,5,7,10], 'RF__n_jobs':[-1]},
    {'RF__min_samples_leaf':[3,5,7,10], 'RF__min_samples_split':[2,3,5,10], 'RF__n_jobs':[-1]},
    {'RF__n_estimators':[100,200], 'RF__max_depth':[6,8,10,12], 'RF__min_samples_split':[2,3,10]},
    ] 

# 2. 모델
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import make_pipeline, Pipeline

# model = make_pipeline(MinMaxScaler(), RandomForestClassifier())
pipe = Pipeline([('minmax', MinMaxScaler()), ('RF', RandomForestClassifier())], verbose=1)

# 3. 훈련
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV

# model = GridSearchCV(RandomForestClassifier(), parameters, cv=5, verbose=1)
model = HalvingGridSearchCV(pipe, parameters, cv=5, verbose=1)

model.fit(x_train, y_train)

# 4. 평가, 예측
result = model.score(x_test, y_test)

print('model.score: ', result)

# model.score:  0.8491620111731844

# 파이프라인 안쓰고
# best tuned acc:  0.9450100255587205
