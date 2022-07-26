from tensorflow.keras.datasets import fashion_mnist
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=0.1,
    shear_range=0.7,
    fill_mode='nearest'
)

augument_size = 10
randidx = np.random.randint(x_train.shape[0], size=augument_size)

x_augument = x_train[randidx].copy()
y_augument = y_train[randidx].copy()
x_10_train = x_train[randidx]

x_augument = x_augument.reshape(x_augument.shape[0], x_augument.shape[1], x_augument.shape[2], 1)
x_10_train = x_10_train.reshape(x_10_train.shape[0], x_10_train.shape[1], x_10_train.shape[2], 1)

x_augument = train_datagen.flow(x_augument, y_augument, batch_size=augument_size, shuffle=False).next()[0]
x_10_train = np.concatenate((x_10_train, x_augument))

import matplotlib.pyplot as plt
plt.figure(figsize=(10,5))
for i in range(20):
    plt.subplot(2,10,i+1)
    plt.axis('off')
    plt.imshow(x_10_train[i])
plt.show()

# 이미지 증폭안시킨거, 시킨거 10장씩 매칭시키기