import numpy as np
from sklearn import datasets
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Input, Dense, LSTM, Conv1D
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
# print(dataset_amo.head) # 앞 다섯개만 보기

feature_cols = ['시가', '고가', '저가', '거래량', '기관', '외국계', '종가']
label_cols = ['종가']

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

def split_xy3(dataset_x, dataset_y, time_steps, y_column):
    x, y = list(), list()
    for i in range(len(dataset_x)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column
        
        if y_end_number > len(dataset_x):
            break
        tmp_x = dataset_x[i:x_end_number]
        tmp_y = dataset_y[x_end_number: y_end_number]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

SIZE = 3
COLSIZE = 3
x1, y1 = split_xy3(dataset_amo[feature_cols], dataset_amo[label_cols], SIZE, COLSIZE)
x2, y2 = split_xy3(dataset_sam[feature_cols], dataset_sam[label_cols], SIZE, COLSIZE)
print(x1, y1) # (1030, 3, 7) (1030, 3, 1)


x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1, x2, y1, test_size=0.2, shuffle=False)

# data 스케일링
scaler = MinMaxScaler()
print(x1_train.shape, x1_test.shape) # (824, 3, 7) (206, 3, 7)
print(x2_train.shape, x2_test.shape) # (824, 3, 7) (206, 3, 7)
print(y_train.shape, y_test.shape) # (824, 3, 7) (206, 3, 7)
x1_train = x1_train.reshape(824*3,7)
x1_train = scaler.fit_transform(x1_train)
x1_test = x1_test.reshape(206*3,7)
x1_test = scaler.transform(x1_test)

x2_train = x2_train.reshape(824*3,7)
x2_train = scaler.fit_transform(x2_train)
x2_test = x2_test.reshape(206*3,7)
x2_test = scaler.transform(x2_test)

# Conv1D에 넣기 위해 3차원화
x1_train = x1_train.reshape(824, 3, 7)
x1_test = x1_test.reshape(206, 3, 7)
x2_train = x2_train.reshape(824, 3, 7)
x2_test = x2_test.reshape(206, 3, 7)

# 2. 모델구성
# 2-1. 모델1
input1 = Input(shape=(3, 7))
conv1 = Conv1D(64, 2, activation='relu')(input1)
lstm1 = LSTM(128, activation='relu')(conv1)
dense1 = Dense(64, activation='relu')(lstm1)
output1 = Dense(32, activation='relu')(dense1)

# 2-2. 모델2
input2 = Input(shape=(3, 7))
conv2 = Conv1D(64, 2, activation='relu')(input2)
lstm2 = LSTM(128, activation='swish')(conv2)
dense2 = Dense(64, activation='relu')(lstm2)
dense3 = Dense(32, activation='relu')(dense2)
output2 = Dense(16, activation='relu')(dense3)

from tensorflow.python.keras.layers import concatenate
merge1 = concatenate([output1, output2])
merge2 = Dense(100, activation='relu')(merge1)
merge3 = Dense(100, name='mg3')(merge2)
last_output = Dense(1)(merge3)

model = Model(inputs=[input1, input2], outputs=[last_output])
model.summary()

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
start_time = time.time()
Es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100, restore_best_weights=True)
fit_log = model.fit([x1_train, x2_train], y_train, epochs=300, batch_size=64, callbacks=[Es], validation_split=0.1)
end_time = time.time()
model.save('./_test/keras46_jongga10.h5')

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

# ./_test/keras46_jongga10.h5
# loss:  56813228.0
# prdict:  [[131076.73]]
# 걸린 시간:  210.3737313747406