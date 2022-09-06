from gc import callbacks
import numpy as np
from torch import dropout
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Input, Dropout, GlobalAveragePooling2D
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import keras
import time

# 1. data
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

x_train = x_train.reshape(50000, 32*32*3)
x_test = x_test.reshape(10000, 32*32*3)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(50000, 32, 32, 3)
x_test = x_test.reshape(10000, 32, 32, 3)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)



# 2. model

activation = 'relu'
drop = 0.2
optimizer = 'adam'

inputs = Input(shape=(32,32,3), name='input')
x = Conv2D(128, (2, 2), activation=activation, padding='valid', name='hidden1')(inputs)
x = Dropout(drop)(x)
x = MaxPooling2D()(x)
x = Conv2D(32, (3, 3), activation=activation, padding='valid', name='hidden3')(x)
x = Dropout(drop)(x) 

# x = Flatten()(x) # (25*25*32) / Flatten의 문제점: 연산량이 너무 많아짐
x = GlobalAveragePooling2D()(x)

x = Dense(256, activation=activation, name='hidden4')(x)
x = Dropout(drop)(x)
x = Dense(128, activation=activation, name='hidden5')(x)
x = Dropout(drop)(x)
outputs = Dense(100, activation='softmax', name='outputs')(x)

model = Model(inputs=inputs, outputs=outputs)

model.summary()

# 3. compilel, fit
model.compile(optimizer=optimizer, metrics=['acc'], loss='categorical_crossentropy')

es = EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto', verbose=1, factor=0.5) # learning rate를 0.5만큼 감축시키겠다

start = time.time()
model.fit(x_train, y_train, epochs=300, validation_split=0.2, batch_size=128, callbacks=[es,reduce_lr])
end = time.time()-start

loss, acc = model.evaluate(x_test,y_test)

print('걸린 시간: ', end)
print('loss: ', loss)
print('acc: ', acc)


# 러닝레이트를 줄여서 핑퐁구간을 줄인다
# 얼리스탑과 함께 사용함. 얼리스탑동안 갱신이 없으면 러닝레이트를 줄이고, 다시 갱신이 없다면 최종 얼리스탑 건다

# 걸린 시간:  764.313619852066
# loss:  2.748807907104492
# acc:  0.3041999936103821