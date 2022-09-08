from keras.models import Model
from keras.layers import Dense, Flatten, Input, GlobalAveragePooling2D
from keras.applications import VGG16
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

x_train = x_train.reshape(50000, 32, 32, 3)
x_test = x_test.reshape(10000, 32, 32, 3)

# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

# 2. model
input1 = Input(shape=(32, 32, 3))
vgg16 = VGG16(include_top=False, input_shape=(32, 32, 3))(input1)
vgg16.trainable=False
glob = GlobalAveragePooling2D()(vgg16)
dense = Dense(64, activation='relu')(glob)
dense = Dense(32)(dense)
output1 = Dense(100, activation='softmax')(dense)

model = Model(inputs=input1, outputs=output1)

model.summary()

# 3. compile, fit
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, metrics=['acc'], loss='sparse_categorical_crossentropy')

es = EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto', verbose=1, factor=0.5)
model.fit(x_train, y_train, epochs=200, validation_split=0.2, batch_size=128, callbacks=[es,reduce_lr])


# 4. evaluate, predict
loss = model.evaluate(x_test, y_test)
y_pred = model.predict(x_test)
acc = accuracy_score(y_test, np.argmax(y_pred, axis=1))

print('loss: ', loss)
print('acc: ', acc)


# loss:  [5.07551383972168, 0.4041999876499176]
# acc:  0.4042
