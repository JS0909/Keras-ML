from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
import numpy as np
import pandas as pd
from sqlalchemy import true #pandas : 엑셀땡겨올때 씀
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Activation, Dense, Conv2D, Flatten, MaxPooling2D, Input, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
from keras.layers import BatchNormalization

#1. 데이터
path = './_data/dacon_shopping/'
train_set = pd.read_csv(path + 'train.csv', # + 명령어는 문자를 앞문자와 더해줌
                        index_col=0) # index_col=n n번째 컬럼을 인덱스로 인식
Weekly_Sales = train_set[['Weekly_Sales']]
print(train_set)
print(train_set.shape) # (6255, 12)

test_set = pd.read_csv(path + 'test.csv', # 예측에서 쓸거임                
                       index_col=0)
print(test_set)
print(test_set.shape) # (180, 11)

print(train_set.columns)
print(train_set.info()) # info 정보출력
print(train_set.describe()) # describe 평균치, 중간값, 최소값 등등 출력

train_set.isnull().sum().sort_values(ascending=False)
test_set.isnull().sum().sort_values(ascending=False)

######## 년, 월 ,일 분리 ############

train_set["day"] = [t.dayofweek for t in pd.DatetimeIndex(train_set.Date)]
train_set["month"] = [t.month for t in pd.DatetimeIndex(train_set.Date)]
train_set['year'] = [t.year for t in pd.DatetimeIndex(train_set.Date)]

test_set["day"] = [t.dayofweek for t in pd.DatetimeIndex(test_set.Date)]
test_set["month"] = [t.month for t in pd.DatetimeIndex(test_set.Date)]
test_set['year'] = [t.year for t in pd.DatetimeIndex(test_set.Date)]

train_set.drop(['Date','Weekly_Sales'],axis=1,inplace=True) # 트레인 세트에서 데이트타임 드랍
test_set.drop(['Date'],axis=1,inplace=True) # 트레인 세트에서 데이트타임 드랍

print(train_set)
print(test_set)
##########################################

# ####################원핫인코더###################

df = pd.concat([train_set, test_set])
print(df)

alldata = pd.get_dummies(df, columns=['day','Store','month', 'year', 'IsHoliday'])
print(alldata)

train_set2 = alldata[:len(train_set)]
test_set2 = alldata[len(train_set):]

print(train_set2)
print(test_set2)
# train_set = pd.get_dummies(train_set, columns=['Store','month', 'year', 'IsHoliday'])
# test_set = pd.get_dummies(test_set, columns=['Store','month', 'year', 'IsHoliday'])




###############프로모션 결측치 처리###############

train_set2 = train_set2.fillna(0)
test_set2 = test_set2.fillna(0)

print(train_set2)
print(test_set2)

##########################################

train_set2 = pd.concat([train_set2, Weekly_Sales],axis=1)
print(train_set2)

x = train_set2.drop(['Weekly_Sales'], axis=1)
y = train_set2['Weekly_Sales']


x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.7,
                                                    random_state=66
                                                    )


scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_set2 = scaler.transform(test_set2)

print(test_set2)


# 2. 모델구성
input1 = Input(shape=(77,))
dense1 = Dense(100)(input1)
batchnorm1 = BatchNormalization()(dense1)
activ1 = Activation('swish')(batchnorm1)
drp4 = Dropout(0.2)(activ1)
dense2 = Dense(100)(drp4)
batchnorm2 = BatchNormalization()(dense2)
activ2 = Activation('swish')(batchnorm2)
drp5 = Dropout(0.2)(activ2)
dense3 = Dense(150)(drp5)
batchnorm3 = BatchNormalization()(dense3)
activ3 = Activation('swish')(batchnorm3)
drp6 = Dropout(0.2)(activ3)
dense4 = Dense(100)(drp6)
batchnorm4 = BatchNormalization()(dense4)
activ4 = Activation('relu')(batchnorm4)
drp7 = Dropout(0.2)(activ4)
output1 = Dense(1)(drp7)
model = Model(inputs=input1, outputs=output1)   


#3. 컴파일, 훈련


model.compile(loss='mse', optimizer='adam', metrics=['mae'])

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

earlyStopping = EarlyStopping(monitor='val_loss', patience=300, mode='auto', verbose=1, 
                              restore_best_weights=True)        

hist = model.fit(x_train, y_train, epochs=10000, batch_size=64,
                 validation_split=0.3,
                 callbacks=[earlyStopping],
                 verbose=1)


#4. 평가, 예측

print("=============================1. 기본 출력=================================")
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

def RMSE(a, b): 
    return np.sqrt(mean_squared_error(a, b))

rmse = RMSE(y_test, y_predict)


from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)

print('loss : ', loss)
print("RMSE : ", rmse)
print('r2스코어 : ', r2)

print(test_set2)

y_submit = model.predict(test_set2)

print(y_submit)
print(y_submit.shape) # (180, 1)

submission_set = pd.read_csv(path + 'submission.csv', # + 명령어는 문자를 앞문자와 더해줌
                             index_col=0) # index_col=n n번째 컬럼을 인덱스로 인식
submission_set['Weekly_Sales'] = y_submit

submission_set.to_csv(path + 'submission.csv', index = True)

# loss :  [11581024256.0, 60665.58984375]
# RMSE :  107615.17241525506
# r2스코어 :  0.9655605086138119

# loss :  [10525473792.0, 59570.9453125]
# RMSE :  102593.73938212715
# r2스코어 :  0.9686994883823303

# loss :  [10323171328.0, 59649.91015625]
# RMSE :  101603.00622509912
# r2스코어 :  0.9693010986384939

# 안결님 코드
