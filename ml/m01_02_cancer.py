import numpy as np
from sklearn import metrics
from sklearn.datasets import load_breast_cancer
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential
from sklearn.model_selection import train_test_split
import time
from sklearn.svm import LinearSVC

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
model = LinearSVC()

# 3. 컴파일, 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
score = model.score(x_test, y_test)
ypred = model.predict(x_test)

print('acc score: ', score)
print('y_pred: ', ypred)

# acc score:  0.7280701754385965