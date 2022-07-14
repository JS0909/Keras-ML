# keras18_gpu_test3 파일의 서머리를 확인해보시오.

from sklearn.datasets import fetch_covtype
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Conv1D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

# 1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (581012, 54) (581012,)
print(np.unique(y, return_counts=True)) # [1 2 3 4 5 6 7]
# (array([1, 2, 3, 4, 5, 6, 7]), array([211840, 283301,  35754,   2747,   9493,  17367,  20510], dtype=int64))

#==========================================pandas.get_dummies===============================================================
y = pd.get_dummies(y)
print(y.shape)
print(y)
#===========================================================================================================================

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8,shuffle=True, random_state=9)
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape, x_test.shape)
x_train = x_train.reshape(464809, 54, 1)
x_test = x_test.reshape(116203, 54, 1)

#2. 모델구성
model = Sequential()
model.add(Conv1D(10, 2, input_shape=(54, 1), activation='relu'))
model.add(Flatten())
model.add(Dense(20))
model.add(Dense(15))
model.add(Dense(30, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(20, activation='relu'))
model.add(Dense(7, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
Es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50, restore_best_weights=True)
log = model.fit(x_train, y_train, epochs=10, batch_size=50, callbacks=[Es], validation_split=0.2)


#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('accuracy : ', acc)

y_predict = model.predict(x_test)
y_predict = tf.argmax(y_predict, axis=1)
y_test = tf.argmax(y_test, axis=1)

acc_sc = accuracy_score(y_test, y_predict)
print('acc스코어 : ', acc_sc)
print('fetchcovtype')

# DNN
# loss :  0.6437365412712097
# acc스코어 :  0.7107303598013821

# LSTM (오래걸려서 나중에 돌리자)

# Conv1D
