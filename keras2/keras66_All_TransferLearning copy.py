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


# 1. data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.reshape(50000, 32*32*3)
x_test = x_test.reshape(10000, 32*32*3)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(50000, 32, 32, 3)
x_test = x_test.reshape(10000, 32, 32, 3)

# 2. model
model_list = [VGG16, VGG19]
# model_list = [ResNet50, ResNet50V2, ResNet101, ResNet101V2, ResNet152, ResNet152V2]
# model_list = [DenseNet121, DenseNet169, DenseNet201]
# model_list = [InceptionV3, InceptionResNetV2]
# model_list = [MobileNet, MobileNetV2, MobileNetV3Small, MobileNetV3Large]
# model_list = [NASNetLarge, NASNetMobile]
# model_list = [EfficientNetB0, EfficientNetB1, EfficientNetB7, Xception]

for model in model_list:
    modelapi = model(include_top=False, input_shape=(32, 32, 3))
    
    modelapi.trainable = False
    # modelapi.trainable = True
    
    model = Sequential()
    model.add(modelapi)
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    
    model.summary()

    # 3. compile, fit
    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['acc'])
    
    es = EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto', verbose=1, factor=0.5)

    model.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=128, callbacks=[es,reduce_lr])

    print('====================================')
    print('모델명: ', modelapi.__class__.__name__)
    print('전체 가중치 개수: ', len(model.weights))
    print('훈련 가중치 개수: ', len(model.trainable_weights))
    
    # 4. evaluate, predict
    loss = model.evaluate(x_test, y_test)
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, np.argmax(y_pred, axis=1))

    print('loss: ', loss)
    print('acc: ', acc)
    
    input('계속하려면 엔터')

######### 결과출력 #########
# 모델명:  VGG16
# 전체 가중치 개수:  30
# 훈련 가중치 개수:  4

# loss:  [1.1767001152038574, 0.597100019454956]
# acc:  0.5971