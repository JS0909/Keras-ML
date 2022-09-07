import numpy as np
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Input, Dropout, GlobalAveragePooling2D
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras.optimizers import Adam
import time

# 1. data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 28*28*1)
x_test = x_test.reshape(10000, 28*28*1)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

# 2. model
activation = 'relu'
drop = 0.2
optimizer = 'adam'

inputs = Input(shape=(28,28,1), name='input')
x = Conv2D(128, (2, 2), activation=activation, padding='valid', name='hidden1')(inputs)
x = Dropout(drop)(x)
x = MaxPooling2D()(x)
x = Conv2D(32, (3, 3), activation=activation, padding='valid', name='hidden3')(x)
x = Dropout(drop)(x) 

x = GlobalAveragePooling2D()(x)

x = Dense(256, activation=activation, name='hidden4')(x)
x = Dropout(drop)(x)
x = Dense(128, activation=activation, name='hidden5')(x)
x = Dropout(drop)(x)
outputs = Dense(10, activation='softmax', name='outputs')(x)

model = Model(inputs=inputs, outputs=outputs)

model.summary()

# 3. compile
es = EarlyStopping(monitor='val_loss', patience=15, mode='min', verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=7, mode='auto', verbose=1, factor=0.5)
tb = TensorBoard(log_dir='D:/study_data/tensorboard_log/_graph', histogram_freq=0,
                 write_graph=True, write_images=True)

learnig_rate = 0.001
optimizer = Adam(learning_rate=learnig_rate)

model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['acc'])

start = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=1, callbacks=[es,reduce_lr,tb], validation_split=0.2)
end = time.time()

loss, acc = model.evaluate(x_test, y_test)

print('leanrnig_rate: ', learnig_rate)
print('loss: ', round(loss, 4))
print('acc: ', round(acc, 4))
print('시간: ', round(end-start, 4))


############## 시각화 ################
import matplotlib.pyplot as plt
plt.figure(figsize=(9,5))

# 1
plt.subplot(2,1,1)
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc='upper right')

# 2
plt.subplot(2,1,2)
plt.plot(hist.history['acc'], marker='.', c='red', label='acc')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_acc')
plt.grid()
plt.title('acc')
plt.ylabel('acc')
plt.xlabel('epochs')
plt.legend(['acc', 'val_acc'])

plt.show()


# tensorboard 실행방법
# cmd> tensorboard --logdir=보드저장경로
# http://127.0.0.1:6006
# http://localhost:6006
