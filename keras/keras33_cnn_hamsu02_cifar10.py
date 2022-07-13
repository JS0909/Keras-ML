from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Input
from keras.datasets import cifar10
import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping
import tensorflow as tf
from sklearn.metrics import accuracy_score
import ssl # 데이터 자동 다운로드, 로딩이 안될 때 사용
ssl._create_default_https_context = ssl._create_unverified_context # 데이터 자동 다운로드, 로딩이 안될 때 사용

# 1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape) # (10000, 32, 32, 3) (10000, 1)

x_train = x_train.reshape(50000, 3072)
x_test = x_test.reshape(10000, 3072)

print(np.unique(y_train, return_counts=True)) 
#(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000], dtype=int64))

from tensorflow.keras.utils import to_categorical # 노란 줄 없애겠다고 tensorflow 빼버리면 이 버전에서는 to_categofical 못쓴다고 나옴;
y_train= to_categorical(y_train)
y_test=to_categorical(y_test)

# 2. 모델구성
# model = Sequential()
# model.add(Dense(64, input_shape=(32*32*3,)))
# model.add(Dense(64, input_shape=(3072,)))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(32))
# model.add(Dense(10, activation='softmax'))

input1 = Input(shape=(32*32*3,))
dense1 = Dense(64)(input1)
dense2 = Dense(64, activation='relu')(dense1)
dense3 = Dense(128, activation='relu')(dense2)
dense4 = Dense(64)(dense3)
dense4 = Dense(32)(dense3)
dense4 = Dense(32, activation='relu')(dense3)
output1 = Dense(10, activation='softmax')(dense4)
model = Model(inputs=input1,outputs=output1)

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
Es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50, restore_best_weights=True)
log = model.fit(x_train, y_train, epochs=1000, batch_size=32, callbacks=[Es], validation_split=0.2)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
y_predict = tf.argmax(y_predict, axis=1)
y_test = tf.argmax(y_test, axis=1)
acc_sc = accuracy_score(y_test, y_predict)
print('loss : ', loss)
print('acc스코어 : ', acc_sc)

# DNN
# loss :  [2.3026514053344727, 0.10000000149011612]
# acc스코어 :  0.1

# 함수
# loss :  [2.302568197250366, 0.10010000318288803]
# acc스코어 :  0.1001