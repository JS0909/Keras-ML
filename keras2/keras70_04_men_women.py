import numpy as np
from keras.applications import VGG19

# 1. 데이터
x_train = np.load('d:/study_data/_save/_npy/keras47_04_train_x.npy')
y_train = np.load('d:/study_data/_save/_npy/keras47_04_train_y.npy')
x_test = np.load('d:/study_data/_save/_npy/keras47_04_test_x.npy')
y_test = np.load('d:/study_data/_save/_npy/keras47_04_test_y.npy')

mypic = np.load('d:/study_data/_save/_npy/keras47_04_mypic.npy')

print(x_train.shape) # (2016, 150, 150, 3)
print(y_train.shape) # (2016,)
print(x_test.shape) # (504, 150, 150, 3)
print(y_test.shape) # (504,)

# 2. 모델
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, GlobalAveragePooling2D

vgg19 = VGG19(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
model = Sequential()
model.add(vgg19)
model.add(Conv2D(64, (2,2), input_shape=(150,150,3), activation='relu'))
model.add(Conv2D(128, (3,3), activation='relu'))
model.add(GlobalAveragePooling2D())
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # 메트릭스에 'acc'해도 됨

log = model.fit(x_train, y_train, epochs=100, batch_size=10, validation_split=0.2) # 배치사이즈 최대로하면 한덩이라서 이렇게 가능


# 그래프
loss = log.history['loss']
accuracy = log.history['accuracy']
val_loss = log.history['val_loss']
val_accuracy = log.history['val_accuracy']

pred = model.predict(mypic)
print('pred: ', pred)

print('loss: ', loss[-1])
print('accuracy: ', accuracy[-1])
print('val_loss: ', val_loss[-1])
print('val_accuracy: ', val_accuracy[-1])

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['font.family'] = 'malgun gothic'
mpl.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(9,6))
plt.plot(log.history['loss'], c='black', label='loss')
plt.plot(log.history['accuracy'], marker='.', c='red', label='accuracy')
plt.plot(log.history['val_loss'], c='blue', label='val_loss')
plt.plot(log.history['val_accuracy'], marker='.', c='green', label='val_accuracy')
plt.grid()
plt.title('여자 남자 구별')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend()
plt.show()

# pred:  [[0.8790028]] = 남자 0, 여자 1, 87퍼로 여자임 ^^
# loss:  0.015426743775606155
# accuracy:  1.0
# val_loss:  0.45826491713523865
# val_accuracy:  0.625

# loss:  1.0981422662734985
# accuracy:  0.33374691009521484
# val_loss:  1.1038367748260498
# val_accuracy:  0.30693069100379944