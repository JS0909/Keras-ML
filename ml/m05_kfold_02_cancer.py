from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict, train_test_split, KFold, cross_val_score


# 1. 데이터
datasets = load_breast_cancer()

x = datasets['data']
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8,shuffle=True, random_state=9)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=99)

#2. 모델구성
from sklearn.svm import SVC
model = SVC()
    
# 3. 4. 컴파일, 훈련, 평가, 예측
# model.fit(x_train, y_train)
score = cross_val_score(model, x_train, y_train, cv=kfold)
y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)
acc = accuracy_score(y_test, y_predict)

print('acc: ', score, '\n cross_val_score: ', round(np.mean(score),4))
print(y_predict)
print('cross_val_predict acc: ', acc)

# acc:  [0.92307692 0.92307692 0.89010989 0.94505495 0.89010989] 
#  cross_val_score:  0.9143
# [1 1 0 1 1 0 0 0 1 0 0 1 1 1 0 1 1 1 1 1 0 1 1 0 1 1 1 1 1 0 0 1 1 1 0 0 0
#  1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 0 1 1 0 1 1 1 1 0 1 1 1 0 1 0 1 1 1 1 1
#  0 1 1 1 1 1 1 1 1 0 1 1 1 1 0 0 1 1 1 1 1 0 1 1 1 0 1 0 1 1 1 1 0 1 1 1 1
#  1 1 0]
# cross_val_predict acc:  0.8859649122807017