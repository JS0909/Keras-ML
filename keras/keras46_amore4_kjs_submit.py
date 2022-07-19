import numpy as np
from sklearn import datasets
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Input, Dense, LSTM, Conv1D
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
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
print(x1.shape, y1.shape) # (1030, 3, 7) (1030, 3, 1)

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
# 3. 컴파일 훈련
model = load_model('./_test/keras46_jongga5.h5')
model.summary()

# 4. 평가, 예측
loss = model.evaluate([x1_test, x2_test], y_test)
predict = model.predict([x1_test, x2_test])
print('loss: ', loss)
print('prdict: ', predict[-1:])

