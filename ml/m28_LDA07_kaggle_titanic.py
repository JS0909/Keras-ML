import numpy as np
import pandas as pd
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, StratifiedKFold,\
    HalvingRandomSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import time
import warnings
warnings.filterwarnings('ignore') # warnig 출력 안함

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
x = np.array(x)
y = np.array(y).reshape(-1, 1) # 벡터로 표시되어 있는 y데이터를 행렬로 전환

from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# PCA 반복문
# for i in range(x.shape[1]):
#     pca = PCA(n_components=i+1)
#     x2 = pca.fit_transform(x)
#     x_train, x_test, y_train, y_test = train_test_split(x2, y, train_size=0.8, random_state=123, shuffle=True)
#     model = XGBClassifier(tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=1)
#     model.fit(x_train, y_train)
#     results = model.score(x_test, y_test)
#     print(i+1, '의 결과: ', results)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=123, shuffle=True)
print(np.unique(y_train, return_counts=True)) # (array([0, 1], dtype=int64), array([435, 277], dtype=int64))
lda = LinearDiscriminantAnalysis(n_components=1)
lda.fit(x_train, y_train) 
x_train = lda.transform(x_train)
x_test = lda.transform(x_test)

# 2. 모델
model = XGBClassifier(tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=1)

# # 3. 훈련
start = time.time()
model.fit(x_train, y_train)
end = time.time()

# # 4. 평가, 예측
results = model.score(x_test, y_test)
print('결과: ', results)
print('걸린 시간: ', end-start)

# 안한거
# 결과:  0.8268156424581006
# 걸린 시간:  0.5060269832611084

# PCA
# 7 의 결과:  0.8100558659217877

# LDA 1
# 결과:  0.8379888268156425
# 걸린 시간:  0.3433079719543457