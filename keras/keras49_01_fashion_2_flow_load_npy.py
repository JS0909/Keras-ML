# 넘파이에서 불러와서 모델 구성
# 성능 비교

import numpy as np

# 1. 데이터
x_train = np.load('d:/study_data/_save/_npy/keras49_01_train_x.npy')
y_train = np.load('d:/study_data/_save/_npy/keras49_01_train_y.npy')
x_test = np.load('d:/study_data/_save/_npy/keras49_01_test_x.npy')
y_test = np.load('d:/study_data/_save/_npy/keras49_01_test_y.npy')

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

log = model.fit(x_train, y_train, epochs=10, batch_size=10, validation_split=0.2)

# 4. 평가, 예측
loss = log.history['loss']
accuracy = log.history['accuracy']

print('loss: ', loss[-1])
print('accuracy: ', accuracy[-1])

# 증폭 전
# loss:  0.1935724914073944
# accuracy:  0.9296875

# 증폭 후
# loss:  0.02717074565589428
# accuracy:  0.990788459777832