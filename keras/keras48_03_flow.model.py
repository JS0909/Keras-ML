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

'''
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
print(x_augument)
print(x_augument.shape) # (40000, 28, 28, 1)

x_train = np.concatenate((x_train, x_augument))
y_train = np.concatenate((y_train, y_augument))
print(x_train.shape, y_train.shape) # (100000, 28, 28, 1) (100000,)
'''
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

y_train= to_categorical(y_train)
y_test=to_categorical(y_test)

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
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # 메트릭스에 'acc'해도 됨

log = model.fit(x_train, y_train, epochs=1, batch_size=20, validation_split=0.2) # 배치사이즈 최대로하면 한덩이라서 이렇게 가능


# 4. 평가, 예측
loss = log.history['loss']
accuracy = log.history['accuracy']
val_loss = log.history['val_loss']
val_accuracy = log.history['val_accuracy']
pred = model.predict(x_test)

print('loss: ', loss[-1])
print('accuracy: ', accuracy[-1])
print('val_loss: ', val_loss[-1])
print('val_accuracy: ', val_accuracy[-1])
# print('pred: ', pred[-1])

y_predict = tf.argmax(pred, axis=1)
y_test = tf.argmax(y_test, axis=1)
acc_sc = accuracy_score(y_test, y_predict)
print('acc_sc: ', acc_sc)

# 증폭 전
# loss:  0.02817673794925213
# accuracy:  0.9955624938011169
# val_loss:  2.3538882732391357
# val_accuracy:  0.8817499876022339

# 증폭 후
# loss: 0.041921503841876984
# accuracy: 0.992900013923645
# val_accuracy: 0.7947499752044678
