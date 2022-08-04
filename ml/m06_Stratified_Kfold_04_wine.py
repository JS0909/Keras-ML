from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict, train_test_split, StratifiedKFold, cross_val_score


# 1. 데이터
datasets = load_wine()

x = datasets['data']
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8,shuffle=True, random_state=9)

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=99)

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

# acc:  [0.68965517 0.72413793 0.75       0.64285714 0.57142857] 
#  cross_val_score:  0.6756
# [0 0 0 0 1 0 1 1 1 1 1 0 0 1 1 0 1 1 0 0 0 0 0 0 0 1 1 1 0 1 0 1 1 0 1 1]
# cross_val_predict acc:  0.75