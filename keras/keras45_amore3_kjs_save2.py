import numpy as np
from sklearn import datasets
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Input, Dense, LSTM, Conv1D, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
import time
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 1. 데이터
path = './_data/test_amore_0718/'
dataset_sam = pd.read_csv(path + '삼성전자220718.csv', thousands=',', encoding='cp949')
dataset_amo = pd.read_csv(path + '아모레220718.csv', thousands=',', encoding='cp949')

dataset_sam = dataset_sam.drop(['전일비','금액(백만)','신용비','외인(수량)','프로그램','외인비'], axis=1)
dataset_amo = dataset_amo.drop(['전일비','금액(백만)','신용비','외인(수량)','프로그램','외인비'], axis=1)

# dataset_amo.info()
# dataset_sam.info()
dataset_sam = dataset_sam.fillna(0)
dataset_amo = dataset_amo.fillna(0)

dataset_sam = dataset_sam.loc[dataset_sam['일자']>="2018/05/04"] # 액면분할 이후 데이터만 사용
dataset_amo = dataset_amo.loc[dataset_amo['일자']>="2018/05/04"] # 삼성의 액면분할 날짜 이후의 행개수에 맞춰줌
print(dataset_amo.shape, dataset_sam.shape) # (1035, 11) (1035, 11)

dataset_sam = dataset_sam.sort_values(by=['일자'], axis=0, ascending=True) # 오름차순 정렬
dataset_amo = dataset_amo.sort_values(by=['일자'], axis=0, ascending=True)
# print(dataset_amo.head) # 앞 다섯개만 보기

feature_cols = ['시가', '고가', '저가','기관','거래량', '외국계', '종가']
# label_cols = ['종가']

dataset_sam = dataset_sam[feature_cols]
dataset_amo = dataset_amo[feature_cols]
dataset_sam = np.array(dataset_sam)
dataset_amo = np.array(dataset_amo)

# 시계열 데이터 만드는 함수
# def split_x(dataset, size):
#     aaa = []
#     for i in range(len(dataset) - size + 1):
#         subset = dataset[i : (i + size)]
#         aaa.append(subset)
#     return np.array(aaa)
# SIZE = 20
# x1 = split_x(dataset_amo[feature_cols], SIZE)
# x2 = split_x(dataset_sam[feature_cols], SIZE)
# y = split_x(dataset_amo[label_cols], SIZE)

def split_xy3(dataset, time_steps, y_column):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column-1
        
        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i:x_end_number, :-1]
        tmp_y = dataset[x_end_number-1: y_end_number, -1]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

SIZE = 3
COLSIZE = 3
x1, y1 = split_xy3(dataset_amo, SIZE, COLSIZE)
x2, y2 = split_xy3(dataset_sam, SIZE, COLSIZE)
print(x1.shape, y1.shape) # (1031, 3, 6) (1031, 3)


x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1, x2, y1, test_size=0.2, shuffle=False)

# data 스케일링
scaler = MinMaxScaler()
print(x1_train.shape, x1_test.shape) # (824, 3, 6) (207, 3, 6)
print(x2_train.shape, x2_test.shape) # (824, 3, 6) (207, 3, 6)
print(y_train.shape, y_test.shape) # (824, 3) (207, 3)
x1_train = x1_train.reshape(824*3,6)
x1_train = scaler.fit_transform(x1_train)
x1_test = x1_test.reshape(207*3,6)
x1_test = scaler.transform(x1_test)

x2_train = x2_train.reshape(824*3,6)
x2_train = scaler.fit_transform(x2_train)
x2_test = x2_test.reshape(207*3,6)
x2_test = scaler.transform(x2_test)

# Conv1D에 넣기 위해 3차원화
x1_train = x1_train.reshape(824, 3, 6)
x1_test = x1_test.reshape(207, 3, 6)
x2_train = x2_train.reshape(824, 3, 6)
x2_test = x2_test.reshape(207, 3, 6)

# 2. 모델구성
# 2-1. 모델1
input1 = Input(shape=(3, 6))
conv1 = Conv1D(128, 2, activation='relu')(input1)
lstm1 = LSTM(128, activation='relu')(conv1)
dense1 = Dense(128, activation='relu')(lstm1)
drop4 = Dropout(0.3)(dense1)
dense2 = Dense(128, activation='relu')(drop4)
drop5 = Dropout(0.3)(dense2)
dense3 = Dense(64, activation='relu')(drop5)
output1 = Dense(64, activation='relu')(dense3)

# 2-2. 모델2
input2 = Input(shape=(3, 6))
conv2 = Conv1D(128, 2, activation='relu')(input2)
lstm2 = LSTM(128, activation='relu')(conv2)
drop1 = Dropout(0.3)(lstm2)
dense4 = Dense(128, activation='relu')(drop1)
drop2 = Dropout(0.3)(dense4)
dense5 = Dense(128, activation='relu')(drop2)
drop3 = Dropout(0.3)(dense5)
output2 = Dense(64, activation='relu')(drop3)

from tensorflow.python.keras.layers import concatenate
merge1 = concatenate([output1, output2])
merge2 = Dense(128)(merge1)
merge3 = Dense(64, name='mg3')(merge2)
last_output = Dense(1)(merge3)

model = Model(inputs=[input1, input2], outputs=[last_output])
model.summary()

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
start_time = time.time()
Es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=1000, restore_best_weights=True)
fit_log = model.fit([x1_train, x2_train], y_train, epochs=500, batch_size=64, callbacks=[Es], validation_split=0.1)
end_time = time.time()
model.save('./_test/keras46_jongga17.h5')

# 4. 평가, 예측
loss = model.evaluate([x1_test, x2_test], y_test)
predict = model.predict([x1_test, x2_test])
print('loss: ', loss)
print('predict: ', predict[-1:])
print('걸린 시간: ', end_time-start_time)

# ./_test/keras46_siga3.h5
# loss:  208862736.0
# prdict:  [[131148.23]]
# 걸린 시간:  479.57821226119995

# ./_test/keras46_jongga1.h5
# loss:  153507344.0
# prdict:  [[132563.53]]
# 걸린 시간:  1382.7434787750244

# ./_test/keras46_jongga2.h5
# loss:  161115696.0
# prdict:  [[128316.78]]
# 걸린 시간:  1343.3939101696014

# ./_test/keras46_jongga3.h5
# loss:  193656768.0
# prdict:  [[129210.37]]
# 걸린 시간:  1061.091631412506

# ./_test/keras46_jongga4.h5
# loss:  169433376.0
# prdict:  [[127012.805]]
# 걸린 시간:  945.789487361908

# ./_test/keras46_jongga5.h5
# loss:  186838480.0
# prdict:  [[126969.49]]
# 걸린 시간:  946.1726791858673

# ./_test/keras46_jongga6.h5
# loss:  13254822.0
# prdict:  [[134408.97]]
# 걸린 시간:  215.71982312202454

# ./_test/keras46_jongga7.h5
# loss:  71318632.0
# prdict:  [[133562.36]]
# 걸린 시간:  211.7583031654358

# ./_test/keras46_jongga8.h5
# loss:  61163492.0
# prdict:  [[131521.94]]
# 걸린 시간:  210.019136428833

# ./_test/keras46_jongga9.h5
# loss:  63438068.0
# prdict:  [[132596.52]]
# 걸린 시간:  215.62591195106506

# ./_test/keras46_jongga10.h5
# loss:  61402748.0
# prdict:  [[135873.56]]
# 걸린 시간:  216.03136539459229

# ==========스플릿 제대로 한 다음꺼===============

# ./_test/keras46_jongga10.h5
# loss:  56813228.0
# prdict:  [[131076.73]]
# 걸린 시간:  210.3737313747406

# ./_test/keras46_jongga11.h5
# loss:  26977560.0
# prdict:  [[134259.23]]
# 걸린 시간:  180.47482657432556

# ./_test/keras46_jongga13.h5
# loss:  29735624.0
# prdict:  [[136755.34]]
# 걸린 시간:  232.13961672782898


#=====등락률, 개인 추가

# ./_test/keras46_jongga12.h5
# loss:  63092864.0
# prdict:  [[135816.75]]
# 걸린 시간:  74.58813786506653

# ./_test/keras46_jongga14.h5
# loss:  22197126.0
# prdict:  [[135838.55]]
# 걸린 시간:  270.89007449150085

#=====feature_cols = ['기관', '외국계', '시가', '고가', '저가', '거래량', '종가']
# ./_test/keras46_jongga15.h5
# loss:  74431192.0
# prdict:  [[134133.2]]
# 걸린 시간:  39.851489782333374

# feature_cols = ['시가', '고가', '저가','기관','거래량', '외국계', '종가']
# ./_test/keras46_jongga16.h5
# loss:  34319656.0
# prdict:  [[133793.05]]
# 걸린 시간:  365.45223665237427

# ./_test/keras46_jongga17.h5
