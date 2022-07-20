import numpy as np
from keras.preprocessing.image import ImageDataGenerator

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
    fill_mode='nearest')

test_datagen = ImageDataGenerator(
    rescale=1./255
)

xy_train = train_datagen.flow_from_directory(
    'd:/_data/image/brain/train/',
    target_size=(200, 200),
    batch_size=200,
    class_mode='binary',
    color_mode='grayscale',
    shuffle=True
)
  
xy_test = test_datagen.flow_from_directory(
    'd:/_data/image/brain/test/',
    target_size=(200, 200),
    batch_size=200,
    class_mode='binary',
    color_mode='grayscale',
    shuffle=True
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
model.add(Conv2D(64, (2,2), input_shape=(200,200,1), activation='relu'))
model.add(Conv2D(128, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # 메트릭스에 'acc'해도 됨

log = model.fit(xy_train[0][0], xy_train[0][1], epochs=50, validation_split=0.2, batch_size=13) # 배치사이즈 최대로하면 한덩이라서 이렇게 가능
# log = model.fit_generator(xy_train, epochs= 200, validation_data=xy_test, 
#                     steps_per_epoch=32,
#                     validation_steps=4
#                     )

# 그래프
loss = log.history['loss']
accuracy = log.history['accuracy']
val_loss = log.history['val_loss']
val_accuracy = log.history['val_accuracy']

print('loss: ', loss[-1])
print('accuracy: ', accuracy[-1])
print('val_loss: ', val_loss[-1])
print('val_accuracy: ', val_accuracy[-1])

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['font.family'] = 'malgun gothic'
mpl.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(9,6))
plt.plot(log.history['loss'], c='black', label='loss')
plt.plot(log.history['accuracy'], marker='.', c='red', label='accuracy')
plt.plot(log.history['val_loss'], c='blue', label='val_loss')
plt.plot(log.history['val_accuracy'], marker='.', c='green', label='val_accuracy')
plt.grid()
plt.title('뇌 사진')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend()
plt.show()