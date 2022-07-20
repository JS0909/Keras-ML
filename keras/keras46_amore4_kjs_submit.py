import numpy as np
from sklearn import datasets
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Input, Dense, LSTM, Conv1D, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 1. 데이터
path = './_data/test_amore_0718/'
dataset_sam = pd.read_csv(path + '삼성전자220718.csv', thousands=',', encoding='cp949')
dataset_amo = pd.read_csv(path + '아모레220718.csv', thousands=',', encoding='cp949')

dataset_sam = dataset_sam.drop(['전일비','금액(백만)','신용비','외인(수량)','프로그램','외인비'], axis=1)
dataset_amo = dataset_amo.drop(['전일비','금액(백만)','신용비','외인(수량)','프로그램','외인비'], axis=1)

dataset_sam = dataset_sam.fillna(0)
dataset_amo = dataset_amo.fillna(0)

dataset_sam = dataset_sam.loc[dataset_sam['일자']>="2018/05/04"] # 액면분할 이후 데이터만 사용
dataset_amo = dataset_amo.loc[dataset_amo['일자']>="2018/05/04"] # 삼성의 액면분할 날짜 이후의 행개수에 맞춰줌

dataset_sam = dataset_sam.sort_values(by=['일자'], axis=0, ascending=True) # 오름차순 정렬
dataset_amo = dataset_amo.sort_values(by=['일자'], axis=0, ascending=True)

feature_cols = ['기관', '외국계', '시가', '고가', '저가', '거래량', '종가']

dataset_sam = dataset_sam[feature_cols]
dataset_amo = dataset_amo[feature_cols]
dataset_sam = np.array(dataset_sam)
dataset_amo = np.array(dataset_amo)

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

x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1, x2, y1, test_size=0.2, shuffle=False)

# data 스케일링
scaler = MinMaxScaler()
x1_train = x1_train.reshape(824*3,6)
x1_train = scaler.fit_transform(x1_train)
x1_test = x1_test.reshape(207*3,6)
x1_test = scaler.transform(x1_test)

x2_train = x2_train.reshape(824*3,6)
x2_train = scaler.fit_transform(x2_train)
x2_test = x2_test.reshape(207*3,6)
x2_test = scaler.transform(x2_test)

x1_train = x1_train.reshape(824, 3, 6)
x1_test = x1_test.reshape(207, 3, 6)
x2_train = x2_train.reshape(824, 3, 6)
x2_test = x2_test.reshape(207, 3, 6)

# 2. 모델구성
# 3. 컴파일, 훈련
model = load_model('./_test/keras46_jongga19.h5')
model.summary()

# 4. 평가, 예측
predict = model.predict([x1_test, x2_test])
print('predict: ', predict[-1:])

# predict:  [[132645.62]]