import numpy as np
from sklearn import datasets
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, GRU, Conv1D, Flatten, Reshape
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
import time
import datetime as dt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 1. 데이터
path = './_data/stock_exam/'
dataset_sam = pd.read_csv(path + '삼성전자220718.csv', encoding='cp949')
dataset_amo = pd.read_csv(path + '아모레220718.csv',  encoding='cp949')

dataset_sam['일자'] = pd.to_datetime(dataset_sam['일자'], format='%Y/%m/%d')
dataset_sam['연도']=dataset_sam['일자'].dt.year

dataset_amo['일자'] = pd.to_datetime(dataset_sam['일자'], format='%Y/%m/%d')
dataset_amo['연도']=dataset_sam['일자'].dt.year

'''
# 거래량 시각화, 확인
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(16, 9))
sns.lineplot(y=dataset_sam['종가'], x=dataset_sam['일자'])
plt.xlabel('time')
plt.ylabel('price')
plt.show()
'''

dataset_amo.sort_index(ascending=False).reset_index(drop=True)

scaler = MinMaxScaler()
scale_cols = ['시가', '고가', '저가', '종가', '거래량', '기관', '외국계']
df_scaled = scaler.fit_transform(dataset_amo[scale_cols])
df_scaled = pd.DataFrame(df_scaled)
df_scaled.columns = scale_cols