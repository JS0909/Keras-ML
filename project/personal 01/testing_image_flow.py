import numpy as np
import tensorflow as tf
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
tf.random.set_seed(9)

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

testing_img = scale_datagen.flow_from_directory(
    'D:/study_data/_testing_image/',
    target_size=(150, 150),
    batch_size=8000,
    class_mode='categorical',
    shuffle=True
)

np.save('d:/study_data/_save/_npy/_project/testing_img.npy', arr =testing_img[0][0])
