# 오토인코더 : 원래 노드에서 작은 노드를 거쳐 다시 원래 크기로

import numpy as np
from keras.datasets import mnist

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000, 784).astype('float32')/255.
x_test = x_test.reshape(10000, 784).astype('float32')/255.

from keras.models import Sequential, Model
from keras.layers import Dense, Input

input_img = Input(shape=(784,))
encoded = Dense(64, activation='relu')(input_img) # 한번 줄임으로써 중요한 특성만 남겨놓는다

decoded = Dense(784, activation='sigmoid')(encoded) # 데이터 스케일링해서 255를 했기때문에 0~1 사이니까 시그모이드 쓴거

autoencoder = Model(input_img, decoded)

# autoencoder.summary()

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.fit(x_train, x_train, epochs=30, batch_size=128, validation_split=0.2) # 준지도학습 : x로 x를 훈련 시킴








