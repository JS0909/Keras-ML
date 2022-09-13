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
# encoded = Dense(1064, activation='relu')(input_img) # 노드를 늘릴 경우?
# encoded = Dense(16, activation='relu')(input_img) # 많이 줄일 수록 특성이 많이 없어짐. 더 많이 두루뭉술해짐. 형태는 유지됨
# encoded = Dense(1, activation='relu')(input_img)

decoded = Dense(784, activation='sigmoid')(encoded) # 데이터 스케일링해서 255를 했기때문에 0~1 사이니까 시그모이드 쓴거
# decoded = Dense(784, activation='relu')(encoded) # 좋지 않다
# decoded = Dense(784)(encoded) # 좋지 않다
# decoded = Dense(784, activation='tanh')(encoded) # 이걸 쓰기도 하지만 -1~1 사이에 쓰기땜에 지금 스케일링이랑 안맞음


autoencoder = Model(input_img, decoded)

# autoencoder.summary()

autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc']) # acc스코어가 굉장히 안좋음. 상대적으로 비교해야함
# autoencoder.compile(optimizer='adam', loss='mse')

autoencoder.fit(x_train, x_train, epochs=30, batch_size=128, validation_split=0.2) # 준지도학습 : x로 x를 훈련 시킴

decoded_imgs = autoencoder.predict(x_test)

import matplotlib.pyplot as plt
n = 10
plt.figure(figsize=(20,4))
for i in range(n):
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    ax = plt.subplot(2, n, i+1+n)
    plt.imshow(decoded_imgs[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()





