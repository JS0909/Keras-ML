import numpy as np
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255, # MinMaxScale 하겠다
    horizontal_flip=True, # 수평 뒤집기
    vertical_flip=True, # 수직 뒤집기
    width_shift_range=0.1, # 가로 이동 범위
    height_shift_range=5, # 세로 이동 범위
    rotation_range=5, # 회전 범위
    zoom_range=1.2, # 확대 범위
    shear_range=0.7, # 기울이기 범위
    fill_mode='nearest' # 채우기 모드는 가장 가까운 거로
) # 위 내용을 랜덤으로 적용해서 수치화로 땡겨옴, 안넣으면 그냥 수치화

test_datagen = ImageDataGenerator(
    rescale=1./255
) # 평가는 증폭하면 안됨. 원래 있던 걸 맞춰야되니까.

xy_train = train_datagen.flow_from_directory( # 경로상의 폴더에 있는 이미지를 가져오겠다
    'd:/_data/image/brain/train/',
    target_size=(150, 150), # 각각의 이미지를 일정한 크기로 불러온다
    batch_size=5, # y값이 5개(0또는 1로), 1000이어도 됨, 그럼 한번에 가져옴,
                  # 만약 이미지 개수가 안나눠떨어지면 뒤에 몇장의 이미지는 가져오지 못하고 버리는 셈
    class_mode='binary', # ad, normal 중에 하나니까. 분류가 여러가지면 categorical, 뒤에 오는게 1임
    color_mode='grayscale', # 흑백화, 이거 안쓰면 디폴트는 컬러임
    shuffle=True,
) # Found 160 images belonging to 2 classes.
  # 총 y 5개의 batch로 자름 160/5 = 32개 데이터
  
xy_test = test_datagen.flow_from_directory(
    'd:/_data/image/brain/test/',
    target_size=(150, 150),
    batch_size=5,
    class_mode='binary',
    shuffle=True,
) # Found 120 images belonging to 2 classes.

print(xy_train)
# <keras.preprocessing.image.DirectoryIterator object at 0x0000015FCEFB28E0>
# sklearn 데이터형식과 같음 ex)load_boston()처럼
# from sklearn.datasets import load_boston
# datasets = load_boston()
# print(datasets)

print(xy_train[0])
print(xy_train[0][0])
print(xy_train[0][0].shape) # (5, 150, 150, 3) grayscale해주면 (5, 150, 150, 1)
print(xy_train[0][1].shape) # (5, )

print(type(xy_train)) # <class 'keras.preprocessing.image.DirectoryIterator'>
print(type(xy_train[0])) # <class 'tuple'>
print(type(xy_train[0][0])) # <class 'numpy.array'>
print(type(xy_train[0][1])) # <class 'numpy.array'>