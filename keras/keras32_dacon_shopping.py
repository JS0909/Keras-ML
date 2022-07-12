from tensorflow.python.keras.models import Sequential, Input, Model
from tensorflow.python.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from keras.layers import BatchNormalization
from sklearn.preprocessing import LabelEncoder

# 1. 데이터
path = './_data/dacon_shopping/'
train_set = pd.read_csv(path+'train.csv', index_col=0) # index_col = n : n번째 칼럼을 인덱스로 인식
test_set = pd.read_csv(path+'test.csv', index_col=0)

print(train_set.info())
print(train_set.isnull().sum())

train_set = train_set.fillna(0)
test_set = test_set.fillna(0)

# Date열에서 년월일 분리 후 Date열 삭제
train_set['Date'] = pd.to_datetime(train_set['Date'])
train_set['year'] = train_set['Date'].dt.strftime('%Y') # object 타입으로 연도 저장됨
train_set['month'] = train_set['Date'].dt.strftime('%m')
train_set['day'] = train_set['Date'].dt.strftime('%d')
train_set = train_set.drop(['Date'], axis=1)

test_set['Date'] = pd.to_datetime(test_set['Date'])
test_set['year'] = test_set['Date'].dt.strftime('%Y')
test_set['month'] = test_set['Date'].dt.strftime('%m')
test_set['day'] = test_set['Date'].dt.strftime('%d')
test_set = test_set.drop(['Date'], axis=1)

# 트레인과 테스트에 있는 연월일들의 종류가 달라서 한꺼번에 인코딩 해야됨
train_test = pd.concat([train_set, test_set])
# Weekly_Sales를 포함한 채로 합치면 라벨 인코딩을 통해 test_set에도 Weekly_Sales 칼럼이 생김
cols = ['month', 'day', 'year']
for col in cols:
    le = LabelEncoder()
    train_test[col] = le.fit_transform(train_test[col])
    
# 다시 train_set과 test_set으로 나눠줌
train_set = train_test[:len(train_set)] # 트레인셋의 길이만큼 짤라 넣음
test_set = train_test[len(train_set):]
test_set = test_set.drop(['Weekly_Sales'], axis=1)
# train_set과 test_set을 찢은 후에 test_set에서 Weekly_Salse를 드랍시켜야됨
#---------------------------------------------------------------------
print(train_set.shape, test_set.shape)

train_set = pd.get_dummies(train_set, columns=['Store'])
test_set = pd.get_dummies(test_set, columns=['Store'])
print(train_set.shape, test_set.shape) # (6255, 58) (180, 57)

x = train_set.drop(['Weekly_Sales'], axis=1)
y = train_set['Weekly_Sales']

print(x.shape,y.shape)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=9)
print(x_train.shape) # (5004, 13)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(x_train.shape, y_train.shape) # (5004, 13) (5004,)


# # 2. 모델 구성
# 시퀀셜
# model = Sequential()
# model.add(Dense(32, input_dim=56))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.1))
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(1))

# 함수형
input1 = Input(shape=(57,))
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
log = model.fit(x_train, y_train, epochs=2000, batch_size=128, callbacks=[Es], validation_split=0.25)
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

# loss:  [229574115328.0, 384410.6875]
# r2:  0.28966727198096065
# rmse:  479138.95858055534

# loss:  [16499713024.0, 70918.6328125]
# r2:  0.9489477113062993
# rmse:  128451.20429720446