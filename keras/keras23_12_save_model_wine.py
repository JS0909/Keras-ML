from sklearn.datasets import load_wine
from sklearn.preprocessing import OneHotEncoder
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from tensorflow.python.keras.callbacks import EarlyStopping
import time

# 1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (178, 13) (178,)
print(np.unique(y, return_counts=True))

#==========================================pandas.get_dummies===============================================================
y = pd.get_dummies(y)
print(y.shape)
print(y)
#===========================================================================================================================

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8,shuffle=True, random_state=9)
# scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler = MaxAbsScaler() 
# scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(np.min(x_train))
print(np.max(x_train))
print(np.min(x_test))
print(np.max(x_test))


                          
#2. 모델구성
# model = Sequential()
# model.add(Dense(80, input_dim=13, activation='relu'))
# model.add(Dense(100))
# model.add(Dense(90))
# model.add(Dense(70, activation='relu'))
# model.add(Dense(50, activation='relu'))
# model.add(Dense(3, activation='softmax'))

input1 = Input(shape=(13,))
dense1 = Dense(80, activation='relu')(input1)
dense2 = Dense(100)(dense1)
dense3 = Dense(90)(dense2)
dense4 = Dense(70, activation='relu')(dense3)
dense5 = Dense(50, activation='relu')(dense4)
output1 = Dense(3, activation='softmax')(dense5)
model = Model(inputs=input1,outputs=output1)

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
Es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50, restore_best_weights=True)
start_time = time.time()
log = model.fit(x_train, y_train, epochs=1000, batch_size=100, callbacks=[Es], validation_split=0.2)
end_time = time.time()
model.save('./_save/keras23_12_save_model_wine.h5')
# model = load_model('./_save/keras23_12_save_model_wine.h5')

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('accuracy : ', acc)

y_predict = model.predict(x_test)
y_predict = tf.argmax(y_predict, axis=1)
y_test = tf.argmax(y_test, axis=1)

acc_sc = accuracy_score(y_test, y_predict)
print('acc스코어 : ', acc_sc)
print('걸린 시간: ', end_time-start_time)

# loss :  0.011018121615052223
# acc스코어 :  1.0
# 걸린 시간:  3.689239025115967