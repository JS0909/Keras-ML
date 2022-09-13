# [실습] 컨볼루션 오토인코더
# CNN 으로 MLP
# UpSampling 이해하고 추가할 것

import numpy as np
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, UpSampling2D

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255.

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Conv2D(filters=hidden_layer_size, kernel_size=(2,2), strides=2, input_shape=(28,28,1), activation='relu'))
    model.add(Dense(units=10, activation='relu'))
    model.add(Dense(units=5, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))
    model.add(UpSampling2D(size=(2, 2), interpolation='nearest'))
    
    model.summary()
    
    return model

model_01 = autoencoder(hidden_layer_size=1)
model_04 = autoencoder(hidden_layer_size=4)
model_16 = autoencoder(hidden_layer_size=16)
model_32 = autoencoder(hidden_layer_size=32)
model_64 = autoencoder(hidden_layer_size=64)
model_154 = autoencoder(hidden_layer_size=154)

print('----------------- node 1개 시작 --------------------')
model_01.compile(optimizer='adam', loss='binary_crossentropy')
model_01.fit(x_train, x_train, epochs=1)

print('----------------- node 4개 시작 --------------------')
model_04.compile(optimizer='adam', loss='binary_crossentropy')
model_04.fit(x_train, x_train, epochs=1)

print('----------------- node 16개 시작 --------------------')
model_16.compile(optimizer='adam', loss='binary_crossentropy')
model_16.fit(x_train, x_train, epochs=1)

print('----------------- node 32개 시작 --------------------')
model_32.compile(optimizer='adam', loss='binary_crossentropy')
model_32.fit(x_train, x_train, epochs=1)

print('----------------- node 64개 시작 --------------------')
model_64.compile(optimizer='adam', loss='binary_crossentropy')
model_64.fit(x_train, x_train, epochs=1)

print('----------------- node 154개 시작 --------------------')
model_154.compile(optimizer='adam', loss='binary_crossentropy')
model_154.fit(x_train, x_train, epochs=1)

output_01 = model_01.predict(x_test)
output_04 = model_04.predict(x_test)
output_16 = model_16.predict(x_test)
output_32 = model_32.predict(x_test)
output_64 = model_64.predict(x_test)
output_154 = model_154.predict(x_test)

import matplotlib.pylab as plt
import random

fig, axes = plt.subplots(7, 5, figsize=(15, 15))
random_imgs = random.sample(range(output_01.shape[0]), 5)
outputs = [x_test, output_01, output_04, output_16, output_32, output_64, output_154]

for row_num, row in enumerate(axes):
    for col_num, ax in enumerate(row):
        ax.imshow(outputs[row_num][random_imgs[col_num]], cmap='gray')
        # ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        
plt.show()