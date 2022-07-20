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
    fill_mode='nearest'
    )

test_datagen = ImageDataGenerator(
    rescale=1./255
)

xy_train = train_datagen.flow_from_directory(
    'd:/study_data/_data/image/cat_dog/training_set/',
    target_size=(150, 150),
    batch_size=50,
    class_mode='binary',
    shuffle=True
)
  
xy_test = test_datagen.flow_from_directory(
    'd:/study_data/_data/image/cat_dog/test_set/',
    target_size=(150, 150),
    batch_size=50,
    class_mode='binary',
    shuffle=True
)

print(xy_train[0])
print(xy_train[0][0])
print(xy_train[0][0].shape)
print(xy_train[0][1].shape)

np.save('d:/study_data/_save/_npy/keras47_01_train_x.npy', arr =xy_train[0][0])
np.save('d:/study_data/_save/_npy/keras47_01_train_y.npy', arr =xy_train[0][1])
np.save('d:/study_data/_save/_npy/keras47_01_test_x.npy', arr =xy_test[0][0])
np.save('d:/study_data/_save/_npy/keras47_01_test_y.npy', arr =xy_test[0][1])