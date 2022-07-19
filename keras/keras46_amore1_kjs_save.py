import numpy as np
from sklearn import datasets
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Input, Dense, LSTM, Conv1D
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
import time
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 1. 데이터
path = './_data/test_amore_0718/'
dataset_sam = pd.read_csv(path + '삼성전자220718.csv', thousands=',', encoding='cp949')
dataset_amo = pd.read_csv(path + '아모레220718.csv', thousands=',', encoding='cp949')

dataset_sam = dataset_sam.drop(['전일비','금액(백만)','신용비','개인','외인(수량)','프로그램','외인비'], axis=1)
dataset_amo = dataset_amo.drop(['전일비','금액(백만)','신용비','개인','외인(수량)','프로그램','외인비'], axis=1)

# dataset_amo.info()
# dataset_sam.info()
dataset_sam = dataset_sam.fillna(0)
dataset_amo = dataset_amo.fillna(0)

dataset_sam = dataset_sam.loc[dataset_sam['일자']>="2018/05/04"] # 액면분할 이후 데이터만 사용
dataset_amo = dataset_amo.loc[dataset_amo['일자']>="2018/05/04"] # 삼성의 액면분할 날짜 이후의 행개수에 맞춰줌
print(dataset_amo.shape, dataset_sam.shape) # (1035, 11) (1035, 11)

dataset_sam = dataset_sam.sort_values(by=['일자'], axis=0, ascending=True) # 오름차순 정렬
dataset_amo = dataset_amo.sort_values(by=['일자'], axis=0, ascending=True)
print(dataset_amo.head) # 앞 다섯개만 보기

feature_cols = ['시가', '고가', '저가', '거래량', '기관', '외국계', '종가']
label_cols = ['시가']

# 시계열 데이터 만드는 함수
def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        aaa.append(subset)
    return np.array(aaa)

SIZE = 20
x1 = split_x(dataset_amo[feature_cols], SIZE)
x2 = split_x(dataset_sam[feature_cols], SIZE)
y = split_x(dataset_amo[label_cols], SIZE)

x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1, x2, y, test_size=0.2, shuffle=False)

# data 스케일링
scaler = MinMaxScaler()
print(x1_train.shape, x1_test.shape) # (812, 20, 7) (204, 20, 7)
print(x2_train.shape, x2_test.shape) # (812, 20, 7) (204, 20, 7)
print(y_train.shape, y_test.shape) # (812, 20, 1) (204, 20, 1)

x1_train = x1_train.reshape(812*20,7)
x1_train = scaler.fit_transform(x1_train)
x1_test = x1_test.reshape(204*20,7)
x1_test = scaler.transform(x1_test)

x2_train = x2_train.reshape(812*20,7)
x2_train = scaler.fit_transform(x2_train)
x2_test = x2_test.reshape(204*20,7)
x2_test = scaler.transform(x2_test)

# Conv1D에 넣기 위해 3차원화
x1_train = x1_train.reshape(812, 20, 7)
x1_test = x1_test.reshape(204, 20, 7)
x2_train = x2_train.reshape(812, 20, 7)
x2_test = x2_test.reshape(204, 20, 7)

# 2. 모델구성
# 2-1. 모델1
input1 = Input(shape=(20, 7))
dense1 = Conv1D(64, 2, activation='relu', name='d1')(input1)
dense2 = LSTM(128, activation='relu', name='d2')(dense1)
dense3 = Dense(64, activation='relu', name='d3')(dense2)
output1 = Dense(32, activation='relu', name='out_d1')(dense3)

# 2-2. 모델2
input2 = Input(shape=(20, 7))
dense11 = Conv1D(64, 2, activation='relu', name='d11')(input2)
dense12 = LSTM(128, activation='swish', name='d12')(dense11)
dense13 = Dense(64, activation='relu', name='d13')(dense12)
dense14 = Dense(32, activation='relu', name='d14')(dense13)
output2 = Dense(16, activation='relu', name='out_d2')(dense14)

from tensorflow.python.keras.layers import concatenate
merge1 = concatenate([output1, output2], name='m1')
merge2 = Dense(100, activation='relu', name='mg2')(merge1)
merge3 = Dense(100, name='mg3')(merge2)
last_output = Dense(1, name='last')(merge3)

model = Model(inputs=[input1, input2], outputs=[last_output])

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
start_time = time.time()
Es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100, restore_best_weights=True)
fit_log = model.fit([x1_train, x2_train], y_train, epochs=500, batch_size=64, callbacks=[Es], validation_split=0.1)
end_time = time.time()
model.save('./_save/keras46_siga3.h5')

# 4. 평가, 예측
loss = model.evaluate([x1_test, x2_test], y_test)
predict = model.predict([x1_test, x2_test])
print('loss: ', loss)
print('prdict: ', predict[-1:]) # 제일 마지막에 나온거 하나 슬라이싱
print('걸린 시간: ', end_time-start_time)

# loss:  179939504.0
# 133839.45
# 걸린 시간:  464.8620798587799