# Dacon 따릉이 문제풀이
import pandas as pd
from pandas import DataFrame 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.python.keras.callbacks import EarlyStopping
import time
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
import numpy as np

# 1. 데이터
path = 'D:\study_data\_data\ddarung/'
train_set = pd.read_csv(path+'train.csv',index_col=0) # index_col = n : n번째 칼럼을 인덱스로 인식
# print(train_set)
# print(train_set.shape) # (1459, 10)

test_set = pd.read_csv(path+'test.csv', index_col=0)
# print(test_set)
# print(test_set.shape) # (715, 9)

### 결측치 처리(일단 제거로 처리) ###
print(train_set.info())
print(train_set.isnull().sum()) # 결측치 전부 더함
# train_set = train_set.dropna() # nan 값(결측치) 열 없앰
train_set = train_set.fillna(0) # 결측치 0으로 채움
print(train_set.isnull().sum()) # 없어졌는지 재확인

x = train_set.drop(['count'], axis=1) # axis = 0은 열방향으로 쭉 한줄(가로로 쭉), 1은 행방향으로 쭉 한줄(세로로 쭉)
y = train_set['count']

print(x.shape, y.shape) # (1328, 9) (1328,)

# trainset과 testset의 분리
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler() 
scaler = RobustScaler()
scaler.fit(x_train)
scaler.fit(test_set)
test_set = scaler.transform(test_set)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성
from sklearn.svm import LinearSVR
model = LinearSVR()

#3. 컴파일, 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
score = model.score(x_test, y_test)
ypred = model.predict(x_test)

print('r2 score: ', score)
print('y_pred: ', ypred)

# 5. 제출 준비
# submission = pd.read_csv(path + 'submission.csv', index_col=0)
# y_submit = model.predict(test_set)
# submission['count'] = y_submit
# submission.to_csv(path + 'submission.csv', index=True)

# r2 score:  0.5387055478544032