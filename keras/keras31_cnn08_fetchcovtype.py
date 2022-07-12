from sklearn.datasets import fetch_covtype
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
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
# Scaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape, x_test.shape) # (464809, 54) (116203, 54)

x_train = x_train.reshape(464809, 9, 6, 1)
x_test = x_test.reshape(116203, 9, 6, 1)

#2. 모델구성
model = Sequential()
model.add(Conv2D(filters=80, kernel_size=(1,1), strides=1, padding='same', input_shape=(9,6,1)))
model.add(MaxPool2D((1,1), padding='same'))
model.add(Conv2D(100, (1,1),padding='valid', activation='swish'))
model.add(Dropout(0.2))
model.add(Conv2D(90, (1,1),padding='same', activation='swish'))
model.add(Dropout(0.1))
model.add(Conv2D(70, (1,1),padding='valid', activation='swish'))
model.add(Flatten())
model.add(Dense(20, activation='relu'))
model.add(Dense(7, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
Es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50, restore_best_weights=True)
log = model.fit(x_train, y_train, epochs=300, batch_size=128, callbacks=[Es], validation_split=0.2)


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
# loss :  0.6437365412712097
# acc스코어 :  0.7107303598013821

# CNN
# loss :  0.4634298086166382
# acc스코어 :  0.8050910905914649