from tensorflow.keras.datasets import fashion_mnist
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    # vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=0.1,
    # shear_range=0.7,
    fill_mode='nearest'
)

print(x_train.shape)
print(x_train.shape[0])
print(x_train.shape[1])
print(x_train.shape[2])

augument_size = 20
randidx = np.random.randint(x_train.shape[0], size=augument_size)

print(randidx)
print(np.min(randidx), np.max(randidx)) 
print(type(randidx)) 

x_augument = x_train[randidx].copy()
y_augument = y_train[randidx].copy()

print(x_augument.shape)
print(y_augument.shape)

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
x_augument = x_augument.reshape(x_augument.shape[0], x_augument.shape[1], x_augument.shape[2], 1)

print('시작')
import time
start = time.time()
x_augument = train_datagen.flow(x_augument, y_augument, 
                                save_to_dir='d:/study_data/_temp/', # flow_from_directory 에도 별도 저장 옵션 사용 가능
                                batch_size=augument_size, shuffle=False).next()[0]
end = time.time()

# x_train = np.concatenate((x_train, x_augument))
# y_train = np.concatenate((y_train, y_augument))

print('걸린 시간: ', round(end - start, 3), '초')