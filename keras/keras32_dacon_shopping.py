from sklearn.datasets import fetch_california_housing
from tensorflow.python.keras.models import Sequential, load_model, Input, Model
from tensorflow.python.keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPool2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from keras.layers import BatchNormalization
import datetime

# 1. 데이터
path = './_data/dacon_shopping/'
train_set = pd.read_csv(path+'train.csv',index_col=0) # index_col = n : n번째 칼럼을 인덱스로 인식
# print(train_set)
# print(train_set.shape) # (6255, 12)

test_set = pd.read_csv(path+'test.csv', index_col=0)
print(test_set)
print(test_set.shape) # (180, 11)

print(train_set.info())
print(train_set.isnull().sum())

train_set = train_set.fillna(0)
test_set = test_set.fillna(0)

# Date열에서 년월일 분리 후 Date열 삭제
train_set['Date'] = pd.to_datetime(train_set['Date'])
train_set['year'] = train_set['Date'].dt.strftime('%Y')
train_set['month'] = train_set['Date'].dt.strftime('%m')
train_set['day'] = train_set['Date'].dt.strftime('%d')
train_set = train_set.drop(['Date'], axis=1)

test_set['Date'] = pd.to_datetime(test_set['Date'])
test_set['year'] = test_set['Date'].dt.strftime('%Y')
test_set['month'] = test_set['Date'].dt.strftime('%m')
test_set['day'] = test_set['Date'].dt.strftime('%d')
test_set = test_set.drop(['Date'], axis=1)

x = train_set.drop(['Weekly_Sales'], axis=1)
y = train_set['Weekly_Sales']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=9)
print(y_train.shape) # (5004,)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(x_train.shape, y_train.shape)

# # 2. 모델 구성
# 시퀀셜
# model = Sequential()
# model.add(Dense(32, input_dim=13))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.1))
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(1))

# 함수형
input1 = Input(shape=(13,))
dense1 = Dense(50,activation='swish')(input1)
batchnorm1 = BatchNormalization()(dense1)
dense2 = Dense(100)(batchnorm1)
drop1 = Dropout(0.3)(dense2)
dense4 = Dense(50, activation='swish')(drop1)
drop2 = Dropout(0.1)(dense4)
batchnorm2 = BatchNormalization()(drop2)
dense5 = Dense(100, activation='swish')(batchnorm2)
drop3 = Dropout(0.2)(dense5)
output1 = Dense(1)(drop3)
model = Model(inputs=input1, outputs=output1)

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
Es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=128, restore_best_weights=True)
log = model.fit(x_train, y_train, epochs=1, batch_size=128, callbacks=[Es], validation_split=0.25)
model.save('./_save/keras32.msw')

# model = load_model('./_save/keras32.msw')

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss: ', loss)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2: ', r2)

def RMSE(y_test, y_predict): # rmse 계산 사용 법
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_predict, y_test)
print('rmse: ', rmse)

# 5. 제출 준비
# submission = pd.read_csv(path + 'submission.csv', index_col=0)
print(test_set) # (180, 13)
print(x_test) # (1251, 13)
test_set = test_set.astype(np.float32)
y_submit = model.predict(test_set) 
# 에러: Failed to convert a NumPy array to a Tensor (Unsupported object type int).
# 변수명 = 변수명.astype(np.float32) 해주면 해결은 됨
print(y_submit.shape)
# submission['Weekly_Sales'] = y_submit
# submission.to_csv(path + 'submission.csv', index=True)

# loss:  [175242412032.0, 312922.40625]
# r2:  0.45777688143409023
# rmse:  418619.62794955465