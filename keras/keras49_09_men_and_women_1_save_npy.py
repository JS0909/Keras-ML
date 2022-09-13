import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# 1. 데이터
train_datagen = ImageDataGenerator(
    # rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=5,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest'
    )

scale_datagen = ImageDataGenerator()

xy_train = scale_datagen.flow_from_directory(
    'd:/study_data/_data/image/men_women/',
    target_size=(150, 150),
    batch_size=5000,
    class_mode='binary',
    shuffle=True
)

mypic = scale_datagen.flow_from_directory(
    'd:/study_data/_data/image/mypic/',
    target_size=(150, 150),
    batch_size=5000,
    class_mode='binary',
    shuffle=True
)

x = xy_train[0][0]
y = xy_train[0][1]
mypic = mypic[0][0]

# 트레인 테스트 스플릿
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)

# 증폭 사이즈만큼 난수 뽑아서
augument_size = 500
randidx = np.random.randint(x_train.shape[0], size=augument_size)
# 각각 인덱스에 난수 넣고 돌려가면서 이미지 저장
x_augument = x_train[randidx].copy()
y_augument = y_train[randidx].copy()

# x 증폭 데이터 담기
x_augument = train_datagen.flow(x_augument, y_augument, batch_size=augument_size, shuffle=False).next()[0]

# 원본train과 증폭train 합치기
x_train = np.concatenate((x_train, x_augument))
y_train = np.concatenate((y_train, y_augument))

np.save('d:/study_data/_save/_npy/keras49_09_train_x.npy', arr =x_train)
np.save('d:/study_data/_save/_npy/keras49_09_train_y.npy', arr =y_train)
np.save('d:/study_data/_save/_npy/keras49_09_test_x.npy', arr =x_test)
np.save('d:/study_data/_save/_npy/keras49_09_test_y.npy', arr =y_test)

np.save('d:/study_data/_save/_npy/keras49_09_test_y.npy', arr =y_test)
