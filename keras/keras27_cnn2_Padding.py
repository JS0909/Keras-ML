from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPool2D

model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(3,3),
                 strides=1,
                 padding='same', # input의 크기 그냥 유지시키려고 씀
                 input_shape=(28, 28, 1))) # (N, 8, 8, 10)
# 처음부터 MaxPooling 안함
model.add(MaxPool2D())
model.add(Conv2D(7, (2,2),padding='valid', # 디폴트
                 activation='relu'))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

# padding : 원래 shape 그대로 유지하고 싶을 때 사용
# 커널 사이즈에 따라 패딩이 달라짐