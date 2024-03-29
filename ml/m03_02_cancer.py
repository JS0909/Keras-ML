import numpy as np
from sklearn import metrics
from sklearn.datasets import load_breast_cancer
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential
from sklearn.model_selection import train_test_split
import time

# 1. 데이터
datasets = load_breast_cancer()
# print(datasets)
# print(datasets.DESCR) // Instances: 569, Attributes: 30
# print(datasets.feature_names)

x = datasets.data # datasets['data']
y = datasets.target # datasets['target'] // key value니까 이렇게도 가능
print(x.shape, y.shape) # (569, 30) (569,)

x_train, x_test, y_train, y_test =  train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)


# 2. 모델구성
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron, LogisticRegression # !논리회귀(분류임)!
from sklearn.neighbors import KNeighborsClassifier # 최근접 이웃
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier # 결정트리를 여러개 랜덤으로 뽑아서 앙상블해서 봄

models = [LinearSVC, SVC, Perceptron, LogisticRegression, KNeighborsClassifier, DecisionTreeClassifier, RandomForestClassifier]

for model in models:
    model = model()
    model_name = str(model).strip('()')
    model.fit(x_train, y_train)
    result = model.score(x_test, y_test)
    print(model_name, '결과: ', result)

# LinearSVC 결과:  0.8508771929824561
# SVC 결과:  0.8947368421052632
# Perceptron 결과:  0.8947368421052632
# LogisticRegression 결과:  0.956140350877193
# KNeighborsClassifier 결과:  0.9210526315789473
# DecisionTreeClassifier 결과:  0.9210526315789473
# RandomForestClassifier 결과:  0.956140350877193