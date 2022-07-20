import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# 1. 데이터
train_datagen = ImageDataGenerator(
    rescale=1./255,
    )


xy_data = train_datagen.flow_from_directory(
    'd:/study_data/_data/image/men_women/',
    target_size=(150, 150),
    batch_size=50,
    class_mode='binary',
    shuffle=True
)

mypic = train_datagen.flow_from_directory(
    'd:/study_data/_data/image/mypic/',
    target_size=(150, 150),
    batch_size=5000,
    class_mode='binary',
    shuffle=True
)

# print(xy_data[0][0].shape) # (2520, 150, 150, 3)
# print(xy_data[0][1].shape) # (2520,)
# print(xy_data[0])

x = xy_data[0][0]
y = xy_data[0][1]
mypic = mypic[0][0]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True)

np.save('d:/study_data/_save/_npy/keras47_04_train_x.npy', arr =x_train)
np.save('d:/study_data/_save/_npy/keras47_04_train_y.npy', arr =y_train)
np.save('d:/study_data/_save/_npy/keras47_04_test_x.npy', arr =x_test)
np.save('d:/study_data/_save/_npy/keras47_04_test_y.npy', arr =y_test)

np.save('d:/study_data/_save/_npy/keras47_04_mypic.npy', arr=mypic)