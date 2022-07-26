from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Input
import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping
import tensorflow as tf
from sklearn.metrics import accuracy_score
import pandas as pd
from tensorflow.python.keras.layers import concatenate, Concatenate

# 1. 데이터
filepath = 'd:/study_data/_save/_npy/_project/'
suffix = '.npy'

x_train = np.load(filepath+'train_x'+suffix)
y1_train = np.load(filepath+'train_y1'+suffix)
y2_train = np.load(filepath+'train_y2'+suffix)

x_test = np.load(filepath+'test_x'+suffix)
y1_test = np.load(filepath+'test_y1'+suffix)
y2_test = np.load(filepath+'test_y2'+suffix)

print(x_train.shape) # (3400, 150, 150, 3)
print(y1_train.shape, y2_train.shape) # (3400, 30) (3400, 4)
print(y1_test.shape, y2_test.shape) # (200, 30) (200, 4)

# from sklearn.preprocessing import OneHotEncoder
# oh = OneHotEncoder()
# y1_train = y1_train.reshape(-1,1)
# y1_test = y1_test.reshape(-1,1)
# oh.fit(y1_train)
# y1_train = oh.transform(y1_train).toarray()
# y1_test = oh.transform(y1_test).toarray()

# 2. 모델구성
# 2-1. input모델
input1 = Input(shape=(150, 150, 3))
conv1 = Conv2D(100,(2,2), padding='same', activation='swish')(input1)
mp1 = MaxPool2D()(conv1)
conv2 = Conv2D(100,(2,2), activation='swish')(mp1)
flat1 = Flatten()(conv2)
dense1 = Dense(100, activation='relu')(flat1)
dense2 = Dense(100, activation='relu')(dense1)
output = Dense(100, activation='relu')(dense2)

# 2-2. output모델1
output1 = Dense(10)(output)
output2 = Dense(10)(output1)
last_output1 = Dense(30, activation='softmax')(output2)

# 2-3. output모델2
output3 = Dense(10)(output)
output4 = Dense(10)(output3)
last_output2 = Dense(4, activation='softmax')(output4)

model = Model(inputs=input1, outputs=[last_output1, last_output2])
model.summary()

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
Es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100, restore_best_weights=True)
log = model.fit(x_train, [y1_train, y2_train], epochs=1, batch_size=100, callbacks=[Es], validation_split=0.2)

#4. 평가, 예측
loss = model.evaluate(x_test, [y1_test, y2_test])

# y1_train = y1_train.reshape(3400*30)
# y1_train = pd.get_dummies(y1_train)

# y1_test = y1_test.reshape(200*30)
# y1_test = pd.get_dummies(y1_test)

# y2_train = y2_train.reshape(3400*4)
# y2_train = pd.get_dummies(y2_train)

# y2_test = y2_test.reshape(200*4)
# y2_test = pd.get_dummies(y2_test)


y1_pred, y2_pred = model.predict(x_test)
y1_pred = tf.argmax(y1_pred, axis=1)
y2_pred = tf.argmax(y2_pred, axis=1)
acc_sc = accuracy_score([y1_test,y1_pred], [y2_test, y2_pred])
print('loss : ', loss)
print('acc스코어 : ', acc_sc)
