from matplotlib.pyplot import axis
from tensorflow.keras.datasets import fashion_mnist
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Dropout, Conv2D, Flatten
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

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

test_datagen = ImageDataGenerator(rescale=1./255)

augument_size = 64
randidx = np.random.randint(x_train.shape[0], size=augument_size)

x_augument = x_train[randidx].copy()
y_augument = y_train[randidx].copy()

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
x_augument = x_augument.reshape(x_augument.shape[0], x_augument.shape[1], x_augument.shape[2], 1)

x_augumented = train_datagen.flow(x_augument, y_augument, batch_size=augument_size, shuffle=False).next()[0]
x_data_all = np.concatenate((x_train, x_augumented))
print(x_data_all.shape) # (60064, 28, 28, 1)
y_data_all = np.concatenate((y_train, y_augument))
print(y_data_all.shape) # (60064,)
xy_train = test_datagen.flow(x_data_all, y_data_all, batch_size=augument_size, shuffle=False)
print(xy_train[0][0].shape) # (64, 28, 28, 1)
xy_test = test_datagen.flow(x_test, y_test, batch_size=augument_size, shuffle=False)

#### 모델 구성 ####
# 성능비교, 증폭 전 후 비교

# 2. 모델
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten

model = Sequential()
model.add(Conv2D(64, (2,2), input_shape=(28,28,1), activation='relu'))
model.add(Conv2D(128, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

# 3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# sparse_categorical_crossentropy 쓰면 원핫 인코딩 필요 없음

log = model.fit_generator(xy_train, epochs= 200, steps_per_epoch=10, validation_steps=4)

# 4. 평가, 예측
loss = log.history['loss']
accuracy = log.history['accuracy']

print('loss: ', loss[-1])
print('accuracy: ', accuracy[-1])

# loss:  0.1935724914073944
# accuracy:  0.9296875