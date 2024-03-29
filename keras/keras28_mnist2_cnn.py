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
# 데이터 순서대로 즉 총 곱셈값이 변하지 않으면 됨 : reshape

x_train = x_train.reshape(60000, 28, 28, 1) # 데이터의 갯수자체는 성능과 큰 상관이 없을 수 있다
x_test = x_test.reshape(10000, 28, 28, 1)
print(x_train.shape)

print(np.unique(y_train, return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949],
#   dtype=int64))
y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)
print(y_train.shape) # (60000, 10) // output 10개 = 라벨값

# [과제] acc 0.98 이상
# convolution 세개 이상

# 2. 모델구성
model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(3,3), strides=1, padding='same', input_shape=(28, 28, 1)))
model.add(MaxPool2D()) # 처음부터 MaxPooling 안함 // 안겹치게 잘라서 큰 수만 빼냄, 전체 크기가 반땡, 자르는 사이즈 변경 가능하긴 함 디폴트는 2x2
model.add(Conv2D(10, (2,2),padding='valid', activation='relu'))
model.add(Conv2D(5, (2,2),padding='same', activation='relu'))
model.add(Conv2D(4, (2,2),padding='valid', activation='relu'))
model.add(Flatten())
model.add(Dense(32, activation='relu')) # 아웃풋 노드 갯수는 항상 맨 뒤에 붙는다
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary() # (None, 28, 28, 64) ... 데이터 갯수 = None

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
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
print('acc스코어 : ', acc_sc)
print('시간: ', end-start)

# loss :  [0.0747639536857605, 0.9764999747276306]
# acc스코어 :  0.9765
# 시간:  152.1847949028015
