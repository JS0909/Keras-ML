from sklearn.datasets import load_wine
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, MaxPool2D, Conv2D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

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

# Scaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape, x_test.shape) # (142, 13) (36, 13)

x_train = x_train.reshape(142, 13, 1, 1)
x_test = x_test.reshape(36, 13, 1, 1)
        
#2. 모델구성
model = Sequential()
model.add(Conv2D(filters=80, kernel_size=(1,1), strides=1, padding='same', input_shape=(13, 1, 1)))
model.add(MaxPool2D((1,1), padding='same'))
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
Es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50, restore_best_weights=True)
log = model.fit(x_train, y_train, epochs=1000, batch_size=100, callbacks=[Es], validation_split=0.2)



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
# loss :  0.7964231967926025
# acc스코어 :  0.7222222222222222

# CNN
# loss :  0.0030472457874566317
# acc스코어 :  1.0