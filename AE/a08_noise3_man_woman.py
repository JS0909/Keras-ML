# [실습] keras47_4 남자 여자에 noise 넣어서
# predict 첫번째: 원본의 노이즈 제거
# 랜덤하게 5개 원본, 수정본

# predict 두번째: 본인 사진 넣어서 // 원본, 수정본

import numpy as np
import matplotlib.pyplot as plt
import random

# 1. 데이터
x_train = np.load('d:/study_data/_save/_npy/keras47_04_train_x.npy')
x_test = np.load('d:/study_data/_save/_npy/keras47_04_test_x.npy')

mypic = np.load('d:/study_data/_save/_npy/keras47_04_mypic.npy')

print(x_train.shape) # (2647, 100, 100, 3)
print(x_test.shape) # (662, 100, 100, 3)
print(mypic.shape) # (1, 100, 100, 3)

x_train = x_train.reshape(2647, 100, 100, 3)
x_test = x_test.reshape(662, 100, 100, 3)
mypic = mypic.reshape(1, 100, 100, 3)

x_train_noised = x_train + np.random.normal(0, 0.1, size=x_train.shape)
x_test_noised = x_test + np.random.normal(0, 0.1, size=x_test.shape)
mypic_noised = mypic + np.random.normal(0, 0.1, size=mypic.shape)

x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1)
x_test_noised = np.clip(x_test_noised, a_min=0, a_max=1)
mypic_noised = np.clip(mypic_noised, a_min=0, a_max=1)

# 2. 모델
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, UpSampling2D

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Conv2D(filters=hidden_layer_size, kernel_size=(2,2), strides=2, input_shape=(100,100,3), activation='relu'))
    model.add(Conv2D(50, (2,2), activation='relu', padding='same'))
    model.add(Conv2D(30, (2,2), activation='relu', padding='same'))
    model.add(UpSampling2D(size=(2,2), interpolation='nearest'))
    model.add(Dense(units=3, activation='sigmoid'))
    
    model.summary()
    
    return model

model = autoencoder(hidden_layer_size=100)

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 

model.fit(x_train_noised, x_train, epochs=10, batch_size=10, validation_split=0.2)

output = model.predict(x_test_noised)
my_pred = model.predict(mypic_noised)

fig, ((ax1, ax2, ax3, ax4, ax5, axm1), (ax6, ax7, ax8, ax9, ax10, axm2), (ax11, ax12, ax13, ax14, ax15, axm3)) = \
    plt.subplots(3, 6, figsize=(20,7))
    
# 이미지 다섯 개를 무작위로 고른다
random_images = random.sample(range(output.shape[0]), 5)

# 원본(입력) 이미지를 맨 위에 그린다
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(100,100,3))
    if i == 0:
        ax.set_ylabel('INPUT', size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
axm1.imshow(mypic[0].reshape(100,100,3)) # 내 이미지
axm1.set_xticks([])
axm1.set_yticks([])
    
# 노이즈 이미지  
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(x_test_noised[random_images[i]].reshape(100,100,3))
    if i == 0:
        ax.set_ylabel('NOISED', size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
axm2.imshow(mypic_noised[0].reshape(100,100,3)) # 내 이미지
axm2.set_xticks([])
axm2.set_yticks([])

# 오토인코더가 출력한 이미지를 아래에 그린다
for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]):
    ax.imshow(output[random_images[i]].reshape(100,100,3))
    if i == 0:
        ax.set_ylabel('OUTPUT', size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
axm3.imshow(my_pred[0].reshape(100,100,3)) # 내 이미지
axm3.set_xticks([])
axm3.set_yticks([])

plt.tight_layout()
plt.show()