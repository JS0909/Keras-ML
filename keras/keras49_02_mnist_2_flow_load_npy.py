# 넘파이에서 불러와서 모델 구성
# 성능 비교

import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPool2D
from tensorflow.python.keras.callbacks import EarlyStopping
import tensorflow as tf
from sklearn.metrics import accuracy_score
import pandas as pd

# 1. 데이터
x_train = np.load('d:/study_data/_save/_npy/keras49_02_train_x.npy')
y_train = np.load('d:/study_data/_save/_npy/keras49_02_train_y.npy')
x_test = np.load('d:/study_data/_save/_npy/keras49_02_test_x.npy')
y_test = np.load('d:/study_data/_save/_npy/keras49_02_test_y.npy')

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

# 3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
Es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100, restore_best_weights=True)
log = model.fit(x_train, y_train, epochs=50, batch_size=100, callbacks=[Es], validation_split=0.2)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
print('loss : ', loss)
print('acc : ', log.history['accuracy'][-1])
print('mnist')

# 증폭 전
# Conv2D
# loss :  [0.06552130728960037, 0.9810000061988831]
# acc스코어 :  0.981

# 증폭 후
# Conv2D
# loss :  [0.12623269855976105, 0.9821000099182129]
# acc :  0.9984297752380371