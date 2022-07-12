# Dacon 따릉이 문제풀이
import pandas as pd
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
import numpy as np

# 1. 데이터
path = './_data/ddarung/'
train_set = pd.read_csv(path+'train.csv',index_col=0) # index_col = n : n번째 칼럼을 인덱스로 인식
# print(train_set)
# print(train_set.shape) # (1459, 10)

test_set = pd.read_csv(path+'test.csv', index_col=0)
# print(test_set)
# print(test_set.shape) # (715, 9)

### 결측치 처리(일단 제거로 처리) ###
print(train_set.info())
print(train_set.isnull().sum()) # 결측치 전부 더함
train_set = train_set.dropna() # nan 값(결측치) 자동으로 0으로 만듦
print(train_set.isnull().sum()) # 없어졌는지 재확인

x = train_set.drop(['count'], axis=1) # axis = 0은 열방향으로 쭉 한줄(가로로 쭉), 1은 행방향으로 쭉 한줄(세로로 쭉)
y = train_set['count']

print(x.shape, y.shape) # (1328, 9) (1328,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)

# 2. 모델 구성
model = Sequential()
model.add(Dense(10, activation='relu', input_dim=9))
model.add(Dense(30
                , activation='relu'
                ))
model.add(Dense(50
                , activation='relu'
                ))
model.add(Dense(20
                , activation='relu'
                ))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M')
filepath = './_ModelCheckPoint/k25/09/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
Es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50, restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1,
                      save_best_only=True, filepath= "".join([filepath, 'k25_',date, '_', filename]))
log = model.fit(x_train, y_train, epochs=200, batch_size=50, callbacks=[Es,mcp], validation_split=0.25)

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss: ', loss)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2: ', r2)

# 5. 제출 준비
# submission = pd.read_csv(path + 'submission.csv', index_col=0)
# y_submit = model.predict(test_set)
# submission['count'] = y_submit
# submission.to_csv(path + 'submission.csv', index=True)
