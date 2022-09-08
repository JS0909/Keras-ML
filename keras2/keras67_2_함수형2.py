from keras.models import Model
from keras.layers import Dense, Flatten, Input, GlobalAveragePooling2D
from keras.applications import InceptionV3, VGG19
from keras.datasets import cifar100
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.optimizers import Adam
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score

# 1. data
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

x_train = x_train.reshape(50000, 32*32*3)
x_test = x_test.reshape(10000, 32*32*3)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(50000, 32,32,3)
x_test = x_test.reshape(10000, 32,32,3) 


# 2. model
base_model = VGG19(weights='imagenet', include_top=False, input_shape=(32,32,3))

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
output1 = Dense(100, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output1)

model.summary()

for layer in base_model.layers:
    layer.trainable = False
# Total params: 22,077,956
# Trainable params: 275,172
# Non-trainable params: 21,802,784


# base_model.trainable = False 
# Total params: 22,077,956
# Trainable params: 275,172
# Non-trainable params: 21,802,784
  
model.summary()

# print(base_model.layers) # 레이어마다 모델명, 주소 나옴

# 3. compile, fit
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, metrics=['acc'], loss='sparse_categorical_crossentropy')

es = EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto', verbose=1, factor=0.5)
model.fit(x_train, y_train, epochs=200, validation_split=0.2, batch_size=128, callbacks=[es,reduce_lr])


# 4. evaluate, predict
loss, acc = model.evaluate(x_test, y_test)
y_pred = model.predict(x_test)
acc = accuracy_score(y_test, np.argmax(y_pred, axis=1))

print('loss: ', loss)
print('acc: ', acc)

# loss:  2.691336154937744
# acc:  0.3437