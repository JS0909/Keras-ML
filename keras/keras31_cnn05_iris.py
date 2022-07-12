from sklearn.datasets import load_iris
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import datetime

# 1. 데이터
datasets = load_iris()
print(datasets.DESCR)

print(datasets.feature_names)
x = datasets['data']
y = datasets['target']

#==========================================pandas.get_dummies===============================================================
y = pd.get_dummies(y)
print(y.shape)
print(y)
#===========================================================================================================================

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8,shuffle=True, random_state=9)

# Scaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape, x_test.shape) # (120, 4) (30, 4)

x_train = x_train.reshape(120, 2, 2, 1)
x_test = x_test.reshape(30, 2, 2, 1)

# 2. 모델구성
# 시퀀셜
model = Sequential()
model.add(Conv2D(filters=80, kernel_size=(2,2), strides=1, padding='same', input_shape=(2, 2, 1)))
model.add(MaxPool2D((2,2), padding='same'))
model.add(Conv2D(100, (1,1),padding='valid', activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(90, (1,1),padding='same', activation='relu'))
model.add(Dropout(0.1))
model.add(Conv2D(70, (1,1),padding='valid', activation='relu'))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(3, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M')
filepath = './_ModelCheckPoint/k25/05/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
Es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50, restore_best_weights=True)
# mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1,
#                       save_best_only=True, filepath= "".join([filepath, 'k25_',date, '_', filename]))
log = model.fit(x_train, y_train, epochs=100, batch_size=100, callbacks=[Es], validation_split=0.2)


#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('accuracy : ', acc)

y_predict = model.predict(x_test)
y_predict = tf.argmax(y_predict, axis=1)
y_test = tf.argmax(y_test, axis=1)

acc_sc = accuracy_score(y_test, y_predict)
print('acc스코어 : ', acc_sc)

# DNN
# loss :  0.0025447658263146877
# acc스코어 :  1.0
# 걸린 시간:  2.3750057220458984

# CNN
# loss :  0.00835611391812563
# accuracy :  1.0
# acc스코어 :  1.0