import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.applications import VGG16

vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

vgg16.trainable=False # 가중치 동결
# vgg16.summary()

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(10))

# model.trainable=False

model.summary()
                                        # trainable:True / VGG:False / model:False
print(len(model.weights))               #       30            30            30
print(len(model.trainable_weights))     #       30             4             0




########################## 2번 소스에서 아래만 추가 ###############################
# print(model.layers)

import pandas as pd
pd.set_option('max_colwidth', -1)
layers = [(layer, layer.name, layer.trainable) for layer in model.layers]
results = pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])

print(results)