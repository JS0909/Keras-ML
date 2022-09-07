# Dacon 따릉이 문제풀이
import pandas as pd
from pandas import DataFrame 
from sklearn.datasets import fetch_california_housing
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.python.keras.callbacks import EarlyStopping
import time
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np

# 1. 데이터
path = 'd:/study_data/_data/ddarung/'
train_set = pd.read_csv(path+'train.csv',index_col=0) # index_col = n : n번째 칼럼을 인덱스로 인식
# print(train_set)
# print(train_set.shape) # (1459, 10)

test_set = pd.read_csv(path+'test.csv', index_col=0)
# print(test_set)
# print(test_set.shape) # (715, 9)

train_set = train_set.fillna(0)
test_set = test_set.fillna(0)

x = train_set.drop(['count'], axis=1)
y = train_set['count']

print(x.shape, y.shape) # (1328, 9) (1328,)

# trainset과 testset의 분리
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)
# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler() 
# scaler = RobustScaler()
scaler.fit(x_train)
scaler.fit(test_set)
test_set = scaler.transform(test_set)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델 구성
model = Sequential()
model.add(Dense(10, activation='relu', input_dim=9))
model.add(Dense(30, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
Es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50, restore_best_weights=True)
start_time = time.time()
log = model.fit(x_train, y_train, epochs=200, batch_size=50, callbacks=[Es], validation_split=0.25)
end_time = time.time()

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss: ', loss)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2: ', r2)
print('걸린 시간: ', end_time-start_time)

# 5. 제출 준비
# submission = pd.read_csv(path + 'submission.csv', index_col=0)
# y_submit = model.predict(test_set)
# submission['count'] = y_submit
# submission.to_csv(path + 'submission.csv', index=True)

# 1. 스케일러 하기 전
# loss:  [3004.434326171875, 39.94355392456055]
# r2:  0.5421933815440603
# 걸린 시간:  3.871631383895874

# 2. MinMax scaler
# loss:  [2359.261474609375, 35.74492645263672]
# r2:  0.6639766280054502
# 걸린 시간:  6.522473573684692

# 3. Standard scaler
# loss:  [2764.33447265625, 37.859642028808594]
# r2:  0.6740949959814919
# 걸린 시간:  6.501195669174194

# 4. MaxAbsScaler
# loss:  [2914.9482421875, 40.28913116455078]
# r2:  0.5581383807006504
# 걸린 시간:  6.865285873413086

# 5. RobustScaler
# loss:  [2725.19873046875, 38.557491302490234]
# r2:  0.6000904226776222
# 걸린 시간:  6.501389265060425

# Standard
# loss:  [2244.907958984375, 34.09609603881836]
# r2:  0.6919902584816643
# 걸린 시간:  26.060454607009888