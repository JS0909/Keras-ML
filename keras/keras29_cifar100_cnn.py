from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPool2D
from keras.datasets import cifar100
import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping
import tensorflow as tf
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# 1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

import matplotlib.pyplot as plt
plt.imshow(x_train[2], 'gray') # 이미지 보여주기
plt.show()

print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape) # (10000, 32, 32, 3) (10000, 1)

x_train = x_train.reshape(50000, 32, 32, 3)
x_test = x_test.reshape(10000, 32, 32, 3)

print(np.unique(y_train, return_counts=True)) 

y_train= to_categorical(y_train)
y_test=to_categorical(y_test)


# 2. 모델구성
model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(3,3), strides=1, padding='same', input_shape=(32, 32, 3)))
model.add(MaxPool2D())
model.add(Conv2D(32, (2,2),padding='valid', activation='relu'))
model.add(Conv2D(32, (3,3),padding='same'))
model.add(Conv2D(64, (2,2),padding='valid', activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(32))
model.add(Dense(100, activation='softmax'))

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
Es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50, restore_best_weights=True)
log = model.fit(x_train, y_train, epochs=1000, batch_size=100, callbacks=[Es], validation_split=0.2)

#4. 평가, 예측
y_predict = model.predict(x_test)
loss = model.evaluate(x_test, y_test)
y_predict = tf.argmax(y_predict, axis=1)
y_test = tf.argmax(y_test, axis=1)
acc_sc = accuracy_score(y_test, y_predict)
print('loss : ', loss)
print('acc스코어 : ', acc_sc)

# loss :  [3.4127721786499023, 0.1898999959230423]
# acc스코어 :  0.1899