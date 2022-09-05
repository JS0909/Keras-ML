from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPool2D
from tensorflow.keras.datasets import mnist
import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping
import tensorflow as tf
from sklearn.metrics import accuracy_score
import pandas as pd
import time

# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,)

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)
print(x_train.shape)

print(np.unique(y_train, return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949],
#   dtype=int64))
y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)
print(y_train.shape) # (60000, 10) // output 10개 = 라벨값


# 2. 모델구성
model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(3,3), strides=1, padding='same', input_shape=(28, 28, 1)))
model.add(MaxPool2D()) 
model.add(Conv2D(10, (2,2),padding='valid', activation='relu'))
model.add(Conv2D(5, (2,2),padding='same', activation='relu'))
model.add(Conv2D(4, (2,2),padding='valid', activation='relu'))
model.add(Flatten())
model.add(Dense(32, activation='relu')) 
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()

# 3. 컴파일, 훈련
from tensorflow.python.keras.optimizer_v2 import adam, adadelta, adagrad, adamax
from tensorflow.python.keras.optimizer_v2 import rmsprop, nadam

learning_rate = 0.001 # defualt = 0.001
optimizer = adagrad.Adagrad(learning_rate=learning_rate)

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
Es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50, restore_best_weights=True)
start = time.time()
log = model.fit(x_train, y_train, epochs=1000, batch_size=100, callbacks=[Es], validation_split=0.2)
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
y_predict = tf.argmax(y_predict, axis=1)
y_test = tf.argmax(y_test, axis=1)
acc_sc = accuracy_score(y_test, y_predict)
print('loss : ', loss)
print('acc스코어 : ', acc_sc, 'optimizer: {0}  lr: {1}'.format(optimizer.__class__.__name__, learning_rate))
print('시간: ', round(end-start, 2))

# loss :  [0.0747639536857605, 0.9764999747276306]
# acc스코어 :  0.9765
# 시간:  152.1847949028015

# loss :  [0.06618057936429977, 0.9807000160217285]
# acc스코어 :  0.9807 optimizer: Adam  lr: 0.001
# 시간:  162.79056692123413


