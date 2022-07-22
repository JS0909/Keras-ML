# 증폭해서 npy에 저장
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
) # 옵션줘서 이미지 수치화할 제너레이터 설정

scale_datagen = ImageDataGenerator(rescale=1./255) # 스케일링만 해주는 제너레이터 설정

augument_size = 5000 # 증폭 사이즈 정의
randidx = np.random.randint(x_train.shape[0], size=augument_size)
# 0~첫번째 쉐입 크기의 범위에서 랜덤 정수를 augument 갯수만큼 빼와서 randidx에 저장

# x와 y 각각 아까 빼온 인덱스용 숫자에 맞춰서 원래 트레인 데이터에서 뽑아내서 저장
x_augument = x_train[randidx].copy()
y_augument = y_train[randidx].copy()

# 이미지데이터는 4차원으로 Conv2D에 써야됨 x 시리즈 전부 리쉐입
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
x_augument = x_augument.reshape(x_augument.shape[0], x_augument.shape[1], x_augument.shape[2], 1)

# x, y 각각 증폭 데이터 담기
x_augument = train_datagen.flow(x_augument, y_augument, batch_size=augument_size, shuffle=False).next()[0]
y_augument = train_datagen.flow(x_augument, y_augument, batch_size=augument_size, shuffle=False).next()[1]
# 여기까지 현재 x_augument, y_augument만 리스케일 상태

# 나머지 데이터들도 리스케일시켜서 사용
x_train = scale_datagen.flow(x_train, y_train, batch_size=x_train.shape[0], shuffle=False).next()[0]
y_train = scale_datagen.flow(x_train, y_train, batch_size=x_train.shape[0], shuffle=False).next()[1]
x_test = scale_datagen.flow(x_test, y_test, batch_size=x_train.shape[0], shuffle=False).next()[0]
y_test = scale_datagen.flow(x_test, y_test, batch_size=x_train.shape[0], shuffle=False).next()[1]

# 원본train과 증폭train 합치기
x_train = np.concatenate((x_train, x_augument))
y_train = np.concatenate((y_train, y_augument))

# 저장
np.save('d:/study_data/_save/_npy/keras49_01_train_x.npy', arr =x_train)
np.save('d:/study_data/_save/_npy/keras49_01_train_y.npy', arr =y_train)
np.save('d:/study_data/_save/_npy/keras49_01_test_x.npy', arr =x_test)
np.save('d:/study_data/_save/_npy/keras49_01_test_y.npy', arr =y_test)