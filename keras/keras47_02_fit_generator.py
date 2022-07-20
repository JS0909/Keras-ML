import numpy as np
from keras.preprocessing.image import ImageDataGenerator

# 1. 데이터
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
    target_size=(100, 100),
    batch_size=5,
    class_mode='binary',
    color_mode='grayscale',
    shuffle=True,
)
  
xy_test = test_datagen.flow_from_directory(
    'd:/_data/image/brain/test/',
    target_size=(100, 100),
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

'''
print(xy_train[0])
print(xy_train[0][0])
print(xy_train[0][0].shape) # (5, 150, 150, 3) grayscale해주면 (5, 150, 150, 1)
print(xy_train[0][1].shape) # (5, )

print(type(xy_train)) # <class 'keras.preprocessing.image.DirectoryIterator'>
print(type(xy_train[0])) # <class 'tuple'>
print(type(xy_train[0][0])) # <class 'numpy.array'>
print(type(xy_train[0][1])) # <class 'numpy.array'>
'''
# 현재 5, 200, 200, 1 짜리 데이터가 32덩어리

# 2. 모델
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten

model = Sequential()
model.add(Conv2D(10, (2,2), input_shape=(100,100,1), activation='relu'))
model.add(Conv2D(10, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # 메트릭스에 'acc'해도 됨

# model.fit(xy_train[0][0], xy_train[0][1]) # 배치사이즈 최대로하면 한덩이라서 이렇게 가능
log = model.fit_generator(xy_train, epochs= 30, validation_data=xy_test, 
                    validation_steps=4, 
                    steps_per_epoch=32 # dataset/batch size = 16-/5 = 32
                                       # 1에포에 배치 몇개를 돌리겠다
                    )

# 그래프
accuracy = log.histroy['accuracy']
val_accuracy = log.history['val_accuracy']
loss = log.history['loss']
val_loss = log.history['val_loss']

print('loss: ', loss[-1])
print('val_loss: ', val_loss[-1])
print('accuracy: ', accuracy[-1])
print('val_accuracy: ', val_accuracy[-1])