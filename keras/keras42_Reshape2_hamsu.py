from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Conv1D, LSTM, Reshape, GRU, Input
from tensorflow.keras.datasets import mnist
import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping
import tensorflow as tf
from sklearn.metrics import accuracy_score
import pandas as pd

# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,)

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)
print(x_train.shape)

print(np.unique(y_train, return_counts=True))
y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)
print(y_train.shape) # (60000, 10)


# 2. 모델구성
# model = Sequential()
# model.add(Conv2D(filters=64, kernel_size=(3,3), strides=1, padding='same', input_shape=(28, 28, 1))) # (None, 28, 28, 64)
# model.add(MaxPooling2D())                            # (None, 14, 14, 64)
# model.add(Conv2D(32, (3,3), activation='relu'))      # (None, 12, 12, 32)
# model.add(Conv2D(7, (3,3), activation='relu'))       # (None, 10, 10, 7)
# model.add(Flatten())                                 # (None, 700)
# model.add(Dense(100, activation='relu'))             # (None, 100)
# model.add(Reshape(target_shape=(100,1)))             # (None, 100, 1)
# model.add(Conv1D(10, 3, padding='valid'))            # (None, 98, 10)
# model.add(LSTM(16))                                  # (None, 16)
# model.add(Dense(32, activation='relu'))              # (None, 32)
# model.add(Dense(10, activation='softmax'))           # (None, 10)
# model.summary()
'''
model = Sequential()
model.add(Conv1D(10, 2, input_shape=(28,28,1))) # (n, 28, 27, 10)
model.add(Reshape(target_shape=(756, 10))) # (n, 756, 10)
model.add(GRU(10)) # (n, 10)
model.add(Dense(16)) # (n, 16)
model.add(Reshape(target_shape=(4, 4, 1))) # (4, 4, 1)
model.add(Conv2D(16, 3)) # (2, 2, 16)
model.add(Reshape(target_shape=(64, 1)))
model.add(Reshape(target_shape=(64,)))
# model.add(Flatten())
model.add(Dense(10))
'''
input1 = Input(shape=(28,28,1))
conv1_1 = Conv1D(10, 2, activation='relu')(input1)
re1 = Reshape(target_shape=(756, 10))(conv1_1)
gru1 = GRU(10, activation='swish')(re1)
dense1 = Dense(16)(gru1)
re2 = Reshape(target_shape=(4, 4, 1))(dense1)
conv2_1 = Conv2D(16, 3)(re2)
re3 = Reshape(target_shape=(8,8))(conv2_1)
re4 = Reshape(target_shape=(64,))(re3)
flatten1 = Flatten()(re4)
output1 = Dense(10)(flatten1)
model = Model(inputs=input1, outputs=output1)
model.summary()

# Model: "model"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# input_1 (InputLayer)         [(None, 28, 28, 1)]       0
# _________________________________________________________________
# conv1d (Conv1D)              (None, 28, 27, 10)        30
# _________________________________________________________________
# reshape (Reshape)            (None, 756, 10)           0
# _________________________________________________________________
# gru (GRU)                    (None, 10)                630
# _________________________________________________________________
# dense (Dense)                (None, 16)                176
# _________________________________________________________________
# reshape_1 (Reshape)          (None, 4, 4, 1)           0
# _________________________________________________________________
# conv2d (Conv2D)              (None, 2, 2, 16)          160
# _________________________________________________________________
# reshape_2 (Reshape)          (None, 8, 8)              0
# _________________________________________________________________
# reshape_3 (Reshape)          (None, 64)                0
# _________________________________________________________________
# flatten (Flatten)            (None, 64)                0
# _________________________________________________________________
# dense_1 (Dense)              (None, 10)                650
# =================================================================
# Total params: 1,646
# Trainable params: 1,646
# Non-trainable params: 0
# _________________________________________________________________


# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
Es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50, restore_best_weights=True)
log = model.fit(x_train, y_train, epochs=1000, batch_size=100, callbacks=[Es], validation_split=0.2)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
y_predict = tf.argmax(y_predict, axis=1)
y_test = tf.argmax(y_test, axis=1)
acc_sc = accuracy_score(y_test, y_predict)
print('loss : ', loss)
print('acc스코어 : ', acc_sc)

# loss :  [0.06552130728960037, 0.9810000061988831]
# acc스코어 :  0.981