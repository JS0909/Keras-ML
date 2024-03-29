import numpy as np
import tensorflow as tf
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
tf.random.set_seed(999)

# 1. 데이터
train_datagen = ImageDataGenerator(
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
    target_size=(224, 224),
    batch_size=5000,
    class_mode='categorical',
    shuffle=True
) # Found 3000 images belonging to 30 classes.

xy2_train = scale_datagen.flow_from_directory(
    'd:/study_data/_data/image/dog/age/',
    target_size=(224, 224),
    batch_size=1000,
    class_mode='categorical',
    shuffle=True
) # Found 951 images belonging to 4 classes.

# print(xy1_train.class_indices)
# print(xy2_train.class_indices)
# {'beagle': 0, 'bichon': 1, 'bulldog': 2, 'chihuahua': 3, 'chow_chow': 4, 
# 'cocker_spaniel': 5, 'collie': 6, 'dachshund': 7, 'fox_terrier': 8, 'german_shepherd': 9, 
# 'golden_retriever': 10, 'greyhound': 11, 'husky': 12, 'jack_russell_terrier': 13, 'jindo': 14, 
# 'labrador_retriever': 15, 'maltese': 16, 'miniature_pinscher': 17, 'papillon': 18, 'pomeranian': 19, 
# 'poodle': 20, 'pug': 21, 'rottweiler': 22, 'samoyed': 23, 'schnauzer': 24, 'shiba': 25,
# 'shihtzu': 26, 'spitz': 27, 'welsh_corgi': 28, 'yorkshire_terrier': 29}
# {'11year_': 0, '5month_4year': 1, '5year_10year': 2, '_4month': 3}

# 파일 불러온 변수에서 xy 분리
x1_train = xy1_train[0][0]
y1_train = xy1_train[0][1]
x2_train = xy2_train[0][0]
y2_train = xy2_train[0][1]

# input 데이터 하나로
x_train = np.concatenate((x1_train, x2_train))

print(x1_train.shape, x2_train.shape) # (3000, 150, 150, 3) (951, 150, 150, 3)
print(x_train.shape, y1_train.shape, y2_train.shape) # (3951, 150, 150, 3) (3000, 30) (951, 4)

# train_test_split을 위한 x, y1, y2 행값 맞춰주기
randidx = np.random.randint(y1_train.shape[0], size=x_train.shape[0]-y1_train.shape[0])
y1_train_aug = y1_train[randidx]
randidx = np.random.randint(y2_train.shape[0], size=x_train.shape[0]-y2_train.shape[0])
y2_train_aug = y2_train[randidx]

y1_train = np.concatenate((y1_train, y1_train_aug))
y2_train = np.concatenate((y2_train, y2_train_aug))

x_train, x_test, y1_train, y1_test, y2_train, y2_test = train_test_split(x_train, y1_train, y2_train, 
                                                        train_size=0.8, shuffle=True, random_state=9)

# 증폭 사이즈만큼 난수 뽑아서
augument_size2 = 1500
randidx = np.random.randint(x_train.shape[0], size=augument_size2)
# 각각 인덱스에 난수 넣고 돌려가면서 이미지 저장
x1_augument = x_train[randidx].copy()
y1_augument = y1_train[randidx].copy() # flow를 위해 생성함

# x_train 증폭 데이터 담기
x_augument = train_datagen.flow(x1_augument, y1_augument, batch_size=augument_size2, shuffle=False).next()[0]

# 원본train과 증폭train 합치기
x_train = np.concatenate((x_train, x_augument))

# 모델에 넣기 위해 x_train, y1_train, y2_train 행 맞추기
randidx = np.random.randint(y1_train.shape[0], size=x_train.shape[0]-y1_train.shape[0])
y1_train_aug = y1_train[randidx]
randidx = np.random.randint(y2_train.shape[0], size=x_train.shape[0]-y2_train.shape[0])
y2_train_aug = y2_train[randidx]
y1_train = np.concatenate((y1_train, y1_train_aug))
y2_train = np.concatenate((y2_train, y2_train_aug))

print(x_train.shape, y1_train.shape, y2_train.shape) # (4660, 150, 150, 3) (4660, 30) (4660, 4)
print(x_test.shape, y1_test.shape, y2_test.shape) # (791, 150, 150, 3) (791, 30) (791, 4)

np.save('d:/study_data/_save/_npy/_project/train_x.npy', arr =x_train)
np.save('d:/study_data/_save/_npy/_project/train_y1.npy', arr =y1_train)
np.save('d:/study_data/_save/_npy/_project/train_y2.npy', arr =y2_train)

np.save('d:/study_data/_save/_npy/_project/test_x.npy', arr =x_test)
np.save('d:/study_data/_save/_npy/_project/test_y1.npy', arr =y1_test)
np.save('d:/study_data/_save/_npy/_project/test_y2.npy', arr =y2_test)
print('데이터 수치화 저장 완료')
