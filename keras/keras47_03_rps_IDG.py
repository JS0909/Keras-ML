import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# 1. 데이터
train_datagen = ImageDataGenerator(
    rescale=1./255,
    # horizontal_flip=True,
    # vertical_flip=True,
    # width_shift_range=0.1,
    # height_shift_range=5,
    # rotation_range=5,
    # zoom_range=1.2,
    # shear_range=0.7,
    # fill_mode='nearest'
    )


xy_data = train_datagen.flow_from_directory(
    'd:/study_data/_data/image/rps/',
    target_size=(150, 150),
    batch_size=5000,
    class_mode='categorical',
    shuffle=True
)

x = xy_data[0][0]
y = xy_data[0][1]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True)

np.save('d:/study_data/_save/_npy/keras47_03_train_x.npy', arr =x_train)
np.save('d:/study_data/_save/_npy/keras47_03_train_y.npy', arr =y_train)
np.save('d:/study_data/_save/_npy/keras47_03_test_x.npy', arr =x_test)
np.save('d:/study_data/_save/_npy/keras47_03_test_y.npy', arr =y_test)