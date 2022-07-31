import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
tf.random.set_seed(9)

# 1. 데이터
scale_datagen = ImageDataGenerator(rescale=1./255)

testing_img = scale_datagen.flow_from_directory(
    'D:/study_data/_testing_image/',
    target_size=(224, 224),
    batch_size=8000,
    class_mode='categorical',
    shuffle=True
)

np.save('d:/study_data/_save/_npy/_project/testing_img.npy', arr =testing_img[0][0])
