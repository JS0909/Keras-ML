from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Conv1D, Flatten
from keras.datasets import cifar100
import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping
import tensorflow as tf
from sklearn.metrics import accuracy_score
import pandas as pd
from tensorflow.keras.utils import to_categorical

# 1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

# import matplotlib.pyplot as plt
# plt.imshow(x_train[2], 'gray') # 이미지 보여주기
# plt.show()

print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape) # (10000, 32, 32, 3) (10000, 1)

x_train = x_train.reshape(50000, 96, 32)
x_test = x_test.reshape(10000, 96, 32)

print(np.unique(y_train, return_counts=True)) 

y_train= to_categorical(y_train)
y_test=to_categorical(y_test)


# 2. 모델구성
model = Sequential()
model.add(Conv1D(80, 2, input_shape=(96,32)))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.3))
model.add(Dense(90, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(70))
model.add(Dropout(0.1))
model.add(Dense(50))
model.add(Dense(100, activation='softmax'))

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
Es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50, restore_best_weights=True)
log = model.fit(x_train, y_train, epochs=1, batch_size=128, callbacks=[Es], validation_split=0.2)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
y_predict = tf.argmax(y_predict, axis=1)
y_test = tf.argmax(y_test, axis=1)
acc_sc = accuracy_score(y_test, y_predict)
print('loss : ', loss)
print('acc스코어 : ', acc_sc)
print('cifar100')

# CNN
# loss :  [4.60683536529541, 0.009999999776482582]
# acc스코어 :  0.01

# LSTM
# loss :  [4.449116230010986, 0.029200000688433647]
# acc스코어 :  0.0292

# Conv1D
# loss :  [4.606482028961182, 0.009999999776482582]
# acc스코어 :  0.01