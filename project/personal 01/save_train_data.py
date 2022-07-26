import numpy as np
import tensorflow as tf
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
    class_mode='categorical',
    shuffle=True
)

xy2_train = scale_datagen.flow_from_directory(
    'd:/study_data/_data/image/dog/age/',
    target_size=(150, 150),
    batch_size=500,
    class_mode='categorical',
    shuffle=True
)

# 파일 불러온 변수에서 xy 분리
x1_train = xy1_train[0][0]
y1_train = xy1_train[0][1]
x2_train = xy2_train[0][0]
y2_train = xy2_train[0][1]

# input 데이터 하나로
x_train = np.concatenate((x1_train, x2_train))

print(x_train.shape, y1_train.shape, y2_train.shape) # (1000, 150, 150, 3) (500, 30) (500, 4)

augument_size1 = 500
randidx1 = np.random.randint(y1_train.shape[0], size=augument_size1)
y1_train_aug = y1_train[randidx1].copy()
randidx1 = np.random.randint(y2_train.shape[0], size=augument_size1)
y2_train_aug = y2_train[randidx1].copy()
# 스플릿을 위한 x y 행값 맞추기
y1_train_aug = train_datagen.flow(x1_train, y1_train_aug, batch_size=augument_size1, shuffle=False).next()[1]
y2_train_aug = train_datagen.flow(x2_train, y2_train_aug, batch_size=augument_size1, shuffle=False).next()[1]

y1_train = np.concatenate((y1_train, y1_train_aug))
y2_train = np.concatenate((y2_train, y2_train_aug))

x_train, x_test, y1_train, y1_test, y2_train, y2_test = train_test_split(x_train, y1_train, y2_train, 
                                                        train_size=0.8, shuffle=True, random_state=9)

# 증폭 사이즈만큼 난수 뽑아서
augument_size2 = 2600
randidx2 = np.random.randint(x_train.shape[0], size=augument_size2)
# 각각 인덱스에 난수 넣고 돌려가면서 이미지 저장
x1_augument = x_train[randidx2].copy()
y1_augument = y1_train[randidx2].copy()
y2_augument = y2_train[randidx2].copy()

# x 증폭 데이터 담기
x_augument = train_datagen.flow(x1_augument, y1_augument, batch_size=augument_size2, shuffle=False).next()[0]

x_train = scale_datagen.flow(x_train, y1_train, batch_size=augument_size2, shuffle=False).next()[0]

# 원본train과 증폭train 합치기
x_train = np.concatenate((x_train, x_augument))
y1_train = np.concatenate((y1_train, y1_augument))
y2_train = np.concatenate((y2_train, y2_augument))

np.save('d:/study_data/_save/_npy/_project/train_x.npy', arr =x_train)
np.save('d:/study_data/_save/_npy/_project/train_y1.npy', arr =y1_train)
np.save('d:/study_data/_save/_npy/_project/train_y2.npy', arr =y2_train)

np.save('d:/study_data/_save/_npy/_project/test_x.npy', arr =x_test)
np.save('d:/study_data/_save/_npy/_project/test_y1.npy', arr =y1_test)
np.save('d:/study_data/_save/_npy/_project/test_y2.npy', arr =y2_test)
print('데이터 수치화 저장 완료')