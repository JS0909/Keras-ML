# 증폭해서 npy에 저장
from re import X
from tensorflow.keras.datasets import cifar100
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    # vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=0.1,
    # shear_range=0.7,
    fill_mode='nearest'
)

scale_datagen = ImageDataGenerator(rescale=1./255)

augument_size = 500
randidx = np.random.randint(x_train.shape[0], size=augument_size)

x_augument = x_train[randidx].copy()
y_augument = y_train[randidx].copy()

# x 증폭 데이터 담기
x_augument = train_datagen.flow(x_augument, y_augument, batch_size=augument_size, shuffle=False).next()[0]
x_train = scale_datagen.flow(x_train, y_train, batch_size=augument_size, shuffle=False).next()[0]

# x 시리즈 전부 리쉐입
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 3)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 3)
x_augument = x_augument.reshape(x_augument.shape[0], x_augument.shape[1], x_augument.shape[2], 3)

# 원본train과 증폭train 합치기
x_train = np.concatenate((x_train, x_augument))
y_train = np.concatenate((y_train, y_augument))

np.save('d:/study_data/_save/_npy/keras49_04_train_x.npy', arr =x_train)
np.save('d:/study_data/_save/_npy/keras49_04_train_y.npy', arr =y_train)
np.save('d:/study_data/_save/_npy/keras49_04_test_x.npy', arr =x_test)
np.save('d:/study_data/_save/_npy/keras49_04_test_y.npy', arr =y_test)
