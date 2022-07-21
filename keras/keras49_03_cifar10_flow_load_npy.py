from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, Input
from keras.datasets import cifar10
import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping
import tensorflow as tf

# 1. 데이터
x_train = np.load('d:/study_data/_save/_npy/keras49_03_train_x.npy')
y_train = np.load('d:/study_data/_save/_npy/keras49_03_train_y.npy')
x_test = np.load('d:/study_data/_save/_npy/keras49_03_test_x.npy')
y_test = np.load('d:/study_data/_save/_npy/keras49_03_test_y.npy')

# 2. 모델
input1 = Input(shape=(32,32,3))
conv1 = Conv2D(10, kernel_size=(3,3), strides=1, padding='same')(input1)
conv2 = Conv2D(30, kernel_size=(2,2), activation='relu')(conv1)
conv3 = Conv2D(30, kernel_size=(1,1), padding='same', activation='relu')(conv2)
conv4 = Conv2D(30, kernel_size=(1,1))(conv3)
conv5 = Conv2D(50, kernel_size=(1,1))(conv4)
flat = Flatten()(conv5)
dense7 = Dense(32, activation='relu')(flat)
output1 = Dense(10, activation='softmax')(dense7)
model = Model(inputs=input1,outputs=output1)

# 3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

log = model.fit(x_train, y_train, epochs=10, batch_size=10, validation_split=0.2)

# 4. 평가, 예측
loss = log.history['loss']
accuracy = log.history['accuracy']

print('loss: ', loss[-1])
print('accuracy: ', accuracy[-1])

# 증폭 전
# loss :  [1.3569039106369019, 0.5206000208854675]
# acc스코어 :  0.5206

# 증폭 후
# loss:  2.302875518798828
# accuracy:  0.1000247523188591