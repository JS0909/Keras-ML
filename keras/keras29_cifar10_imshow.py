from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPool2D
from keras.datasets import cifar10
import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping
import tensorflow as tf
from sklearn.metrics import accuracy_score
import ssl # 데이터 자동 다운로드, 로딩이 안될 때 사용
ssl._create_default_https_context = ssl._create_unverified_context # 데이터 자동 다운로드, 로딩이 안될 때 사용

# 1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

import matplotlib.pyplot as plt
plt.imshow(x_train[2], 'gray') # 이미지 보여주기
plt.show()

print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape) # (10000, 32, 32, 3) (10000, 1)

x_train = x_train.reshape(50000, 32, 32, 3)
x_test = x_test.reshape(10000, 32, 32, 3)

print(np.unique(y_train, return_counts=True)) 
#(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000], dtype=int64))
