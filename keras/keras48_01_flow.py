from tensorflow.keras.datasets import fashion_mnist
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=0.1,
    shear_range=0.7,
    fill_mode='nearest'
)

augument_size = 100

print(x_train[0].shape) # (28, 28)
print(x_train[0].reshape(28*28).shape) # (784,)
print(np.tile(x_train[0].reshape(28*28), augument_size).reshape(-1,28,28,1).shape) # (100, 28, 28, 1)

print(np.zeros(augument_size))
print(np.zeros(augument_size).shape) # (100,)

xy_data = train_datagen.flow(
    np.tile(x_train[0].reshape(28*28), augument_size).reshape(-1,28,28,1), # x
    np.zeros(augument_size),                                               # y
    batch_size=augument_size,
    shuffle=True,
).next() # 쉐이프 하나 건너뛰고 다음꺼

print(xy_data) # <keras.preprocessing.image.NumpyArrayIterator object at 0x000001CE71654A60>
print(xy_data[0]) # 첫 배치의 x와 y
print(xy_data[0][0].shape) # (100, 28, 28, 1).nex() 사용 후 (28, 28, 1)
print(xy_data[0][1].shape) # (100,) .nex() 사용 후 (28, 28, 1)
print(xy_data[0].shape) # (100, 28, 28, 1)
print(xy_data[1].shape) # (100,)


import matplotlib.pyplot as plt
plt.figure(figsize=(7,7))
for i in range(49):
    plt.subplot(7,7,i+1)
    plt.axis('off')
    plt.imshow(xy_data[0][i],cmap='gray') # 첫번째 배치([0])의 x([0])의 i번째([i]) 이미지를 흑백으로 가져와서 보여준다
    # plt.imshow(xy_data[0][0][i],cmap='gray') # .next() 안쓰면 이렇게 쓰면 됨
plt.show()

# 다중분류는 분류별 각각의 훈련 데이터의 갯수가 비슷할 수록 성능이 좋다