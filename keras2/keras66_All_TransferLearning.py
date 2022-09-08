import numpy as np

from keras.applications import VGG16, VGG19
from keras.applications import ResNet50, ResNet50V2, ResNet101, ResNet101V2, ResNet152, ResNet152V2
from keras.applications import DenseNet121, DenseNet169, DenseNet201
from keras.applications import InceptionV3, InceptionResNetV2
from keras.applications import MobileNet, MobileNetV2, MobileNetV3Small, MobileNetV3Large
from keras.applications import NASNetLarge, NASNetMobile
from keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB7
from keras.applications import Xception

from keras.datasets import cifar10
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Flatten
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score


# 2. model
# model_list = [VGG16, VGG19]
# model_list = [ResNet50, ResNet50V2, ResNet101, ResNet101V2, ResNet152, ResNet152V2]
# model_list = [DenseNet121, DenseNet169, DenseNet201]
# model_list = [InceptionV3, InceptionResNetV2]
# model_list = [MobileNet, MobileNetV2, MobileNetV3Small, MobileNetV3Large]
# model_list = [NASNetLarge, NASNetMobile]
model_list = [EfficientNetB0, EfficientNetB1, EfficientNetB7, Xception]

for model in model_list:
    modelapi = model()
    
    # modelapi.trainable = False
    modelapi.trainable = True
    
    modelapi.summary()
    
    input('계속하려면 엔터')

    print('====================================')
    print('모델명: ', modelapi.name)
    print('전체 가중치 개수: ', len(modelapi.weights))
    print('훈련 가중치 개수: ', len(modelapi.trainable_weights))
    


######### 결과출력 #########
# 모델명:  vgg16
# 전체 가중치 개수:  26
# 훈련 가중치 개수:  0      26
# ====================================
# 모델명:  vgg19
# 전체 가중치 개수:  32
# 훈련 가중치 개수:  0      32
# ====================================
# 모델명:  resnet50
# 전체 가중치 개수:  318
# 훈련 가중치 개수:  0      212
# ====================================
# 모델명:  resnet50v2
# 전체 가중치 개수:  270
# 훈련 가중치 개수:  0      172
# ====================================
# 모델명:  resnet101
# 전체 가중치 개수:  624
# 훈련 가중치 개수:  0      416
# ====================================
# 모델명:  resnet101v2
# 전체 가중치 개수:  542
# 훈련 가중치 개수:  0      342
# ====================================
# 모델명:  resnet152
# 전체 가중치 개수:  930
# 훈련 가중치 개수:  0      620
# ====================================
# 모델명:  resnet152v2
# 전체 가중치 개수:  814
# 훈련 가중치 개수:  0      512
# ====================================
# 모델명:  densenet121
# 전체 가중치 개수:  604
# 훈련 가중치 개수:  0      362
# ====================================
# 모델명:  densenet169
# 전체 가중치 개수:  844
# 훈련 가중치 개수:  0      506
# ====================================
# 모델명:  densenet201
# 전체 가중치 개수:  1004
# 훈련 가중치 개수:  0      602
# ====================================
# 모델명:  inception_v3
# 전체 가중치 개수:  378
# 훈련 가중치 개수:  0      190
# ====================================
# 모델명:  inception_resnet_v2
# 전체 가중치 개수:  898
# 훈련 가중치 개수:  0      490
# ====================================
# 모델명:  mobilenet_1.00_224
# 전체 가중치 개수:  137
# 훈련 가중치 개수:  0      83
# ====================================
# 모델명:  mobilenetv2_1.00_224
# 전체 가중치 개수:  262
# 훈련 가중치 개수:  0      158
# ====================================
# 모델명:  MobilenetV3small
# 전체 가중치 개수:  210
# 훈련 가중치 개수:  0      142
# ====================================
# 모델명:  MobilenetV3large
# 전체 가중치 개수:  266
# 훈련 가중치 개수:  0      174
# ====================================
# 모델명:  NASNet
# 전체 가중치 개수:  1546
# 훈련 가중치 개수:  0      1018
# ====================================
# 모델명:  NASNet
# 전체 가중치 개수:  1126
# 훈련 가중치 개수:  0      742
# ====================================
# 모델명:  efficientnetb0
# 전체 가중치 개수:  314
# 훈련 가중치 개수:  0      213
# ====================================
# 모델명:  efficientnetb1
# 전체 가중치 개수:  442
# 훈련 가중치 개수:  0      301
# ====================================
# 모델명:  efficientnetb7
# 전체 가중치 개수:  1040
# 훈련 가중치 개수:  0      711
# ====================================
# 모델명:  xception
# 전체 가중치 개수:  236
# 훈련 가중치 개수:  0      156