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

print(x_train.shape) # (60000, 28, 28)
print(x_train.shape[0]) # 60000
print(x_train.shape[1]) # 28
print(x_train.shape[2]) # 28

augument_size = 40000
randidx = np.random.randint(x_train.shape[0], size=augument_size)
# randint(a, b) : 0 ~ a-1의 범위에서 b개만큼 랜덤으로 뽑음
# randint(a, b, c) : a ~ b-1의 범위에서 c개만큼 랜덤으로 뽑음
print(randidx) # [31836 43069 58347 ... 35389 27138 46941]
print(np.min(randidx), np.max(randidx)) # 1 59997
print(type(randidx)) # <class 'numpy.ndarray'>

x_augument = x_train[randidx].copy() # .copy() : 새로운 메모리를 사용해서 쓰겠다. 안넣어도 되긴 함. 근데 안정적.
y_augument = y_train[randidx].copy()

print(x_augument.shape) # (40000, 28, 28)
print(y_augument.shape) # (40000,)

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
x_augument = x_augument.reshape(x_augument.shape[0], x_augument.shape[1], x_augument.shape[2], 1)

x_augument = train_datagen.flow(x_augument, y_augument, batch_size=augument_size, shuffle=False).next()[0]
# x만 뽑음, 안섞는 이유는 y와 섞을 필요 없어서
# 근데 섞는 옵션 줘도 .next()[0]으로 x만 뱉어서 저장해두니까 섞인 y_augument는 저장 안하니까 상관은 없음, 걍 연산량만 좀 늘어남
print(x_augument)
print(x_augument.shape) # (40000, 28, 28, 1)

x_train = np.concatenate((x_train, x_augument))
y_train = np.concatenate((y_train, y_augument))
print(x_train.shape, y_train.shape) # (100000, 28, 28, 1) (100000,)