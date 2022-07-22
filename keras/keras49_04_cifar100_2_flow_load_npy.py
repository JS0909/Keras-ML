from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv1D, Dropout, Flatten, LSTM
from keras.datasets import cifar10
import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

# 1. 데이터
x_train = np.load('d:/study_data/_save/_npy/keras49_04_train_x.npy')
y_train = np.load('d:/study_data/_save/_npy/keras49_04_train_y.npy')
x_test = np.load('d:/study_data/_save/_npy/keras49_04_test_x.npy')
y_test = np.load('d:/study_data/_save/_npy/keras49_04_test_y.npy')

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*3, x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*3, x_test.shape[2])

# 2. 모델구성
# model = Sequential()
# model.add(Conv1D(80, 2, input_shape=(32,32,3)))
# model.add(Flatten())
# model.add(Dense(100))
# model.add(Dropout(0.3))
# model.add(Dense(90, activation='relu'))
# model.add(Dropout(0.1))
# model.add(Dense(70))
# model.add(Dropout(0.1))
# model.add(Dense(50))
# model.add(Dense(100, activation='softmax'))

model = Sequential()
model.add(LSTM(80, input_shape=(96,32)))
model.add(Dense(100))
model.add(Dropout(0.3))
model.add(Dense(90, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(70))
model.add(Dropout(0.1))
model.add(Dense(50))
model.add(Dense(100, activation='softmax'))

# 3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
Es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50, restore_best_weights=True)
log = model.fit(x_train, y_train, epochs=1, batch_size=128, callbacks=[Es], validation_split=0.2)

# 4. 평가, 예측
loss = log.history['loss']
accuracy = log.history['accuracy']

print('loss: ', loss[-1])
print('accuracy: ', accuracy[-1])

# 증폭 전
# loss :  [4.606482028961182, 0.009999999776482582]
# acc스코어 :  0.01

# 증폭 후
# loss:  4.555176734924316
# accuracy:  0.01792079210281372