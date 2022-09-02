# Dacon 따릉이 문제풀이
import pandas as pd
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Conv1D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 1. 데이터
path = 'D:/study_data/_data/ddarung/'
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

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape, x_test.shape)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

# 2. 모델 구성
model = Sequential()
model.add(Conv1D(10, 2, activation='relu', input_shape=(9,1)))
model.add(Flatten())
model.add(Dense(30, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(20, activation='relu'))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
Es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50, restore_best_weights=True)
log = model.fit(x_train, y_train, epochs=200, batch_size=50, callbacks=[Es], validation_split=0.25)

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss: ', loss)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2: ', r2)
print('따릉이')

# 5. 제출 준비
test_set = test_set.values.reshape(test_set.shape[0], test_set.shape[1], 1)
submission = pd.read_csv(path + 'submission.csv', index_col=0)
y_submit = model.predict(test_set)
submission['count'] = y_submit
submission.to_csv(path + 'submission.csv', index=True)

# DNN
# loss:  [4022.582275390625, 43.64305877685547]
# r2:  0.5567107693605131

# LSTM
# loss:  [2494.2431640625, 36.75603485107422]
# r2:  0.5989452585466708

# Conv1D
# loss:  [1841.220703125, 31.671733856201172]
# r2:  0.7134192579447114