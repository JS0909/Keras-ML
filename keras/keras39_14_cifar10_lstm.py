from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, LSTM
from keras.datasets import cifar10
import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping
import tensorflow as tf
from sklearn.metrics import accuracy_score
import ssl # 데이터 자동 다운로드, 로딩이 안될 때 사용
ssl._create_default_https_context = ssl._create_unverified_context # 데이터 자동 다운로드, 로딩이 안될 때 사용

# 1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape, x_test.shape) # (50000, 32, 32, 3) (10000, 32, 32, 3)

x_train = x_train.reshape(50000, 96, 32)
x_test = x_test.reshape(10000, 96, 32)

from tensorflow.keras.utils import to_categorical # 노란 줄 없애겠다고 tensorflow 빼버리면 이 버전에서는 to_categofical 못쓴다고 나옴;
y_train= to_categorical(y_train)
y_test=to_categorical(y_test)


# 2. 모델구성
model = Sequential()
model.add(LSTM(80, input_shape=(96,32)))
model.add(Dense(100))
model.add(Dropout(0.3))
model.add(Dense(90, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(70))
model.add(Dropout(0.1))
model.add(Dense(50))
model.add(Dense(10, activation='softmax'))

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
Es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50, restore_best_weights=True)
log = model.fit(x_train, y_train, epochs=5, batch_size=100, callbacks=[Es], validation_split=0.2)

#4. 평가, 예측
y_predict = model.predict(x_test)
loss = model.evaluate(x_test, y_test)
y_predict = tf.argmax(y_predict, axis=1)
y_test = tf.argmax(y_test, axis=1)
acc_sc = accuracy_score(y_test, y_predict)
print('loss : ', loss)
print('acc스코어 : ', acc_sc)

# CNN
# loss :  [1.3569039106369019, 0.5206000208854675]
# acc스코어 :  0.5206

# LSTM
# loss :  [2.0952553749084473, 0.22849999368190765]
# acc스코어 :  0.2285