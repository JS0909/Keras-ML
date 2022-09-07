import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from keras.applications import VGG16

model = VGG16() # 디폴트 : include_top=True, input_dim=(224, 224, 3)

model.summary()

print(len(model.weights))               # 32
print(len(model.trainable_weights))     # 32

############### include_top = True #################
# 1. FC layer 원래꺼 그대로 쓴다.
# 2. input_shape=(224, 224, 3) 고정값 - 바꿀 수 없다.

# input_1 (InputLayer)        [(None, 224, 224, 3)]     0
# block1_conv1 (Conv2D)       (None, 224, 224, 64)      1792
# ......
# flatten (Flatten)           (None, 25088)             0
# fc1 (Dense)                 (None, 4096)              102764544
# fc2 (Dense)                 (None, 4096)              16781312
# predictions (Dense)         (None, 1000)              4097000
# =================================================================
# Total params: 138,357,544
# Trainable params: 138,357,544
# Non-trainable params: 0




#===================================================================================================================
model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

model.summary()

print(len(model.weights))               # 26
print(len(model.trainable_weights))     # 26

############### include_top = False #################
# 1. FC layer 없어짐 - 커스터마이징 가능
# 2. input_shape=(32, 32, 3) - 바꿀 수 있다.

# input_2 (InputLayer)        [(None, 32, 32, 3)]       0
# block1_conv1 (Conv2D)       (None, 32, 32, 64)        1792
# ......
# block5_conv3 (Conv2D)       (None, 2, 2, 512)         2359808
# block5_pool (MaxPooling2D)  (None, 1, 1, 512)         0
# flatten 하단 없어짐, Fully connected layer 하단 전체가 사라짐
# =================================================================
# Total params: 14,714,688
# Trainable params: 14,714,688
# Non-trainable params: 0