# Kaggle Bike_sharing
from operator import index
import numpy as np
import pandas as pd 
from pandas import DataFrame 
from sklearn.datasets import fetch_california_housing
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.python.keras.callbacks import EarlyStopping


# 1. 데이터
path = './_data/kaggle_bike/'
train_set = pd.read_csv(path+'train.csv')
# print(train_set)
# print(train_set.shape) # (10886, 11)

test_set = pd.read_csv(path+'test.csv')
# print(test_set)
# print(test_set.shape) # (6493, 8)

# datetime 열 내용을 각각 년월일시간날짜로 분리시켜 새 열들로 생성 후 원래 있던 datetime 열을 통째로 drop
train_set["hour"] = [t.hour for t in pd.DatetimeIndex(train_set.datetime)]
train_set["day"] = [t.dayofweek for t in pd.DatetimeIndex(train_set.datetime)]
train_set["month"] = [t.month for t in pd.DatetimeIndex(train_set.datetime)]
train_set['year'] = [t.year for t in pd.DatetimeIndex(train_set.datetime)]
train_set['year'] = train_set['year'].map({2011:0, 2012:1})

test_set["hour"] = [t.hour for t in pd.DatetimeIndex(test_set.datetime)]
test_set["day"] = [t.dayofweek for t in pd.DatetimeIndex(test_set.datetime)]
test_set["month"] = [t.month for t in pd.DatetimeIndex(test_set.datetime)]
test_set['year'] = [t.year for t in pd.DatetimeIndex(test_set.datetime)]
test_set['year'] = test_set['year'].map({2011:0, 2012:1})

train_set.drop('datetime',axis=1,inplace=True) # train_set에서 데이트타임 드랍
test_set.drop('datetime',axis=1,inplace=True) # test_set에서 데이트타임 드랍
train_set.drop('casual',axis=1,inplace=True) # casul 드랍 이유 모르겠음
train_set.drop('registered',axis=1,inplace=True) # registered 드랍 이유 모르겠음

#print(train_set.info())
# null값이 없으므로 결측치 삭제과정 생략

x = train_set.drop(['count'], axis=1)
y = train_set['count']

print(x.shape, y.shape) # (10886, 14) (10886,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)


# 2. 모델 구성
model = Sequential()
model.add(Dense(10, input_dim=12))
model.add(Dense(30, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(30))
model.add(Dense(50, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
Es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50, restore_best_weights=True)
log = model.fit(x_train, y_train, epochs=1000, batch_size=50, callbacks=[Es], validation_split=0.25)

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss: ', loss)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2: ', r2)

'''
#======그래프======
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

mpl.rcParams['font.family'] = 'malgun gothic'
mpl.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(9,6))
plt.plot(log.history['loss'], marker='.', c='red', label='loss')
plt.plot(log.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()
plt.title('캐글 바이크//로스와 발리데이션 로스')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend()
plt.show()
#======그래프======
'''

# 5. 제출 준비
submission = pd.read_csv(path + 'submission.csv', index_col=0)
y_submit = model.predict(test_set)
submission['count'] = np.abs(y_submit) # 마이너스 나오는거 절대값 처리

submission.to_csv(path + 'submission.csv', index=True)

# linear only
# loss:  [mse: 21339.32421875, mae: 109.42781066894531]
# r2:  0.3916957657818575

# with 2 relu
# loss:  [mse: 2070.903076171875, mae: 30.01389503479004]
# r2:  0.9353254005377757

# with 3 relu
# loss:  [mse: 1768.896484375, mae: 26.64974021911621]
# r2:  0.9469869764493036

# with 4 relu
# loss:  [mse: 2206.21435546875, mae: 30.686269760131836]
# r2:  0.9347932518442085

# relu를 세번정도 섞어줬더니 성능이 비약적으로 좋아지는 것을 확인했다