from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

# 1. 데이터
datasets = load_iris()
x = datasets['data']
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, random_state=99)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=99)
                      
#2. 모델구성
from sklearn.svm import SVC
model = SVC()
    
# 3. 4. 컴파일, 훈련, 평가, 예측
# model.fit(x_train, y_train)
score = cross_val_score(model, x_train, y_train, cv=kfold)
# score = cross_val_score(model, x, y, cv=5)
# 이렇게 하면 위에서 kFold로 따로 정의하지 않아도 된다 대신 파라미터들을 건들 수 있는게 줄어듦
print('acc: ', score, '\n cross_val_score: ', round(np.mean(score),4))

y_predict = cross_val_score(model, x_test, y_test, cv=kfold)
print(y_predict)

acc = accuracy_score(y_test, y_predict)
print('cross_val_predict acc: ', acc)

# acc:  [0.95652174 1.         1.         1.         0.95454545]
#  cross_val_score:  0.9822
# [0.875 1.    0.875 1.    1.   ]