# 넘파이 불러와서 모델링
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

# 1. 데이터
x_train = np.load('d:/study_data/_save/_npy/keras49_08_train_x.npy')
y_train = np.load('d:/study_data/_save/_npy/keras49_08_train_y.npy')
x_test = np.load('d:/study_data/_save/_npy/keras49_08_test_x.npy')
y_test = np.load('d:/study_data/_save/_npy/keras49_08_test_y.npy')

# 2. 모델
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten

model = Sequential()
model.add(Conv2D(64, (2,2), input_shape=(150,150,3), activation='relu'))
model.add(Conv2D(128, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='softmax'))

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

log = model.fit(x_train, y_train, epochs=10, batch_size=10, validation_split=0.2)

# 4. 평가, 예측
loss = log.history['loss']
accuracy = log.history['accuracy']
val_loss = log.history['val_loss']
val_accuracy = log.history['val_accuracy']

print('loss: ', loss[-1])
print('accuracy: ', accuracy[-1])
print('val_loss: ', val_loss[-1])
print('val_accuracy: ', val_accuracy[-1])

# 그래프
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
plt.title('가위바위보')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend()
plt.show()

# 증폭 전
# loss:  5.068951395514887e-06
# accuracy:  1.0
# val_loss:  0.0004104298132006079
# val_accuracy:  1.0

# 증폭 후
# loss:  6.067371487006312e-06
# accuracy:  1.0
# val_loss:  2.0046746730804443
# val_accuracy:  0.3452380895614624