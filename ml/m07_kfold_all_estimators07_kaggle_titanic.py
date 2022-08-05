from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.utils import all_estimators
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
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
y = np.array(y).reshape(-1, 1) # 벡터로 표시되어 있는 y데이터를 행렬로 전환


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8,shuffle=True, random_state=9)
         
from sklearn.preprocessing import MinMaxScaler
scaler= MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=99)
         
#2. 모델구성
allAlgorithms = all_estimators(type_filter='classifier')
print('allAlgorithms: ', allAlgorithms)
print('모델의 개수: ', len(allAlgorithms)) # 41

for (name, algorithm) in allAlgorithms:
    try:
        model = algorithm()
        ypred = cross_val_predict(model, x_test, y_test, cv=kfold)
        score = cross_val_score(model, x_train, y_train, cv=kfold)
        print(name, '의 정답률: ', round(np.mean(score),4))
        
    except:
        # continue # 또는 pass
        print(name, '은 안나온 놈')

# AdaBoostClassifier 의 정답률:  0.8132
# BaggingClassifier 의 정답률:  0.7908
# BernoulliNB 의 정답률:  0.7852
# CalibratedClassifierCV 의 정답률:  0.8048
# CategoricalNB 은 안나온 놈
# ClassifierChain 은 안나온 놈
# ComplementNB 의 정답률:  0.7837
# DecisionTreeClassifier 의 정답률:  0.7612
# DummyClassifier 의 정답률:  0.6305
# ExtraTreeClassifier 의 정답률:  0.7921
# ExtraTreesClassifier 의 정답률:  0.7963
# GaussianNB 의 정답률:  0.7964
# GaussianProcessClassifier 의 정답률:  0.8146
# GradientBoostingClassifier 의 정답률:  0.816
# HistGradientBoostingClassifier 의 정답률:  0.8314
# KNeighborsClassifier 의 정답률:  0.7935
# LabelPropagation 의 정답률:  0.8188
# LabelSpreading 의 정답률:  0.8188
# LinearDiscriminantAnalysis 의 정답률:  0.809
# LinearSVC 의 정답률:  0.8062
# LogisticRegression 의 정답률:  0.8076
# LogisticRegressionCV 의 정답률:  0.8076
# MLPClassifier 의 정답률:  0.8174
# MultiOutputClassifier 은 안나온 놈
# MultinomialNB 의 정답률:  0.8076
# NearestCentroid 의 정답률:  0.7978
# NuSVC 의 정답률:  0.816
# OneVsOneClassifier 은 안나온 놈
# OneVsRestClassifier 은 안나온 놈
# OutputCodeClassifier 은 안나온 놈
# PassiveAggressiveClassifier 의 정답률:  0.683
# Perceptron 의 정답률:  0.7654
# QuadraticDiscriminantAnalysis 의 정답률:  0.8103
# RadiusNeighborsClassifier 의 정답률:  0.8034
# RandomForestClassifier 의 정답률:  0.809
# RidgeClassifier 의 정답률:  0.809
# RidgeClassifierCV 의 정답률:  0.809
# SGDClassifier 의 정답률:  0.7949
# SVC 의 정답률:  0.8188
# StackingClassifier 은 안나온 놈
# VotingClassifier 은 안나온 놈