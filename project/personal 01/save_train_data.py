from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPool2D
import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping
import tensorflow as tf
from sklearn.metrics import accuracy_score
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# 1. 데이터
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=5,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest'
    )

scale_datagen = ImageDataGenerator(rescale=1./255)

xy1_train = scale_datagen.flow_from_directory(
    'd:/study_data/_data/image/dog/breed/',
    target_size=(150, 150),
    batch_size=500,
    class_mode='binary',
    color_mode='grayscale',
    shuffle=True
)

xy2_train = scale_datagen.flow_from_directory(
    'd:/study_data/_data/image/dog/age/',
    target_size=(150, 150),
    batch_size=500,
    class_mode='binary',
    color_mode='grayscale',
    shuffle=True
)

x1_train = xy1_train[0][0]
y1_train = xy1_train[0][1]
x2_train = xy2_train[0][0]
y2_train = xy2_train[0][1]

x1_train, x1_test, x2_train, x2_test, y1_train, y1_test, y2_train, y2_test = train_test_split(x1_train, x2_train, y1_train, y2_train, 
                                                        train_size=0.8, shuffle=True, random_state=9)

# 증폭 사이즈만큼 난수 뽑아서
augument_size = 500
randidx = np.random.randint(x1_train.shape[0], size=augument_size)
# 각각 인덱스에 난수 넣고 돌려가면서 이미지 저장
x1_augument = x1_train[randidx].copy()
y1_augument = y1_train[randidx].copy()
x2_augument = x2_train[randidx].copy()
y2_augument = y2_train[randidx].copy()

# x 증폭 데이터 담기
x1_augument = train_datagen.flow(x1_augument, y1_augument, batch_size=augument_size, shuffle=False).next()[0]
x2_augument = train_datagen.flow(x2_augument, y2_augument, batch_size=augument_size, shuffle=False).next()[0]

x1_train = scale_datagen.flow(x1_train, y1_train, batch_size=augument_size, shuffle=False).next()[0]
x2_train = scale_datagen.flow(x2_train, y2_train, batch_size=augument_size, shuffle=False).next()[0]

# 원본train과 증폭train 합치기
x1_train = np.concatenate((x1_train, x1_augument))
y1_train = np.concatenate((y1_train, y1_augument))
x2_train = np.concatenate((x2_train, x2_augument))
y2_train = np.concatenate((y2_train, y2_augument))

np.save('d:/study_data/project/_save/train_x1.npy', arr =x1_train)
np.save('d:/study_data/project/_save/train_y1.npy', arr =y1_train)
np.save('d:/study_data/project/_save/train_x2.npy', arr =x2_train)
np.save('d:/study_data/project/_save/train_y2.npy', arr =y2_train)

np.save('d:/study_data/project/_save/test_x1.npy', arr =x1_test)
np.save('d:/study_data/project/_save/test_y1.npy', arr =y1_test)
np.save('d:/study_data/project/_save/test_x2.npy', arr =x2_test)
np.save('d:/study_data/project/_save/test_y2.npy', arr =y2_test)
print('데이터 수치화 저장 완료')