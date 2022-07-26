from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Input
import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping
import tensorflow as tf
from sklearn.metrics import accuracy_score
import pandas as pd

# 1. 데이터
x1_train = np.load('d:/study_data/project/_save/train_x1.npy')
y1_train = np.load('d:/study_data/project/_save/train_y1.npy')
x2_train = np.load('d:/study_data/project/_save/train_x2.npy')
y2_train = np.load('d:/study_data/project/_save/train_y2.npy')

x1_test = np.load('d:/study_data/project/_save/test_x1.npy')
y1_test = np.load('d:/study_data/project/_save/test_y1.npy')
x2_test = np.load('d:/study_data/project/_save/test_x2.npy')
y2_test = np.load('d:/study_data/project/_save/test_y2.npy')

print(x1_train.shape, y1_train.shape)
print(x2_train.shape, y2_train.shape)
print(x1_test.shape, y1_test.shape)
print(x2_test.shape, y2_test.shape)

# x_train = x_train.reshape(60000, 28, 28, 1) # 데이터의 갯수자체는 성능과 큰 상관이 없을 수 있다
# x_test = x_test.reshape(10000, 28, 28, 1)

# print(np.unique(y_train, return_counts=True))

y1_train = pd.get_dummies(y1_train)
y1_test = pd.get_dummies(y1_test)
y2_train = pd.get_dummies(y2_train)
y2_test = pd.get_dummies(y2_test)

# 2. 모델구성 ////////////////// 하는 중
model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(3,3), strides=1, padding='same', input_shape=(28, 28, 1)))
model.add(MaxPool2D()) # 처음부터 MaxPooling 안함 // 안겹치게 잘라서 큰 수만 빼냄, 전체 크기가 반땡, 자르는 사이즈 변경 가능하긴 함 디폴트는 2x2
model.add(Conv2D(10, (2,2),padding='valid', activation='relu'))
model.add(Conv2D(5, (2,2),padding='same', activation='relu'))
model.add(Conv2D(4, (2,2),padding='valid', activation='relu'))
model.add(Flatten())
model.add(Dense(32, activation='relu')) # 아웃풋 노드 갯수는 항상 맨 뒤에 붙는다
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary() # (None, 28, 28, 64) ... 데이터 갯수 = None

# 2-1. 모델1
input1 = Input(shape=(2,))
dense1 = Dense(100, activation='relu', name='d1')(input1)
dense2 = Dense(100, activation='relu', name='d2')(dense1)
dense3 = Dense(100, activation='relu', name='d3')(dense2)
output1 = Dense(100, activation='relu', name='out_d1')(dense3)

# 2-2. 모델2
input2 = Input(shape=(3,))
dense11 = Dense(100, activation='relu', name='d11')(input2)
dense12 = Dense(100, activation='swish', name='d12')(dense11)
dense13 = Dense(100, activation='relu', name='d13')(dense12)
dense14 = Dense(100, activation='relu', name='d14')(dense13)
output2 = Dense(100, activation='relu', name='out_d2')(dense14)

# 2-3. 모델3
input3 = Input(shape=(2,))
dense15 = Dense(100, activation='relu', name='d15')(input3)
dense16 = Dense(100, activation='swish', name='d16')(dense15)
dense17 = Dense(100, activation='relu', name='d17')(dense16)
dense18 = Dense(100, activation='relu', name='d18')(dense17)
output3 = Dense(100, activation='relu', name='out3')(dense18)

# Concatenate
from tensorflow.python.keras.layers import concatenate, Concatenate
# merge1 = concatenate([output1, output2, output3], name='m1')
merge1 = Concatenate(axis=0)([output1, output2, output3])
merge2 = Dense(100, activation='relu', name='mg2')(merge1)
merge3 = Dense(100, name='mg3')(merge2)
last_output = Dense(1, name='last1')(merge3)

# 2-4. output모델1
output41 = Dense(10)(last_output)
output42 = Dense(10)(output41)
last_output1 = Dense(1)(output42)

# 2-5. output모델2
output51 = Dense(10)(last_output)
output52 = Dense(10)(output51)
last_output2 = Dense(1)(output52)

model = Model(inputs=[input1, input2, input3], outputs=[last_output1, last_output2])

model.summary()







# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
Es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50, restore_best_weights=True)
log = model.fit(x_train, y_train, epochs=1000, batch_size=100, callbacks=[Es], validation_split=0.2)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
y_predict = tf.argmax(y_predict, axis=1)
y_test = tf.argmax(y_test, axis=1)
acc_sc = accuracy_score(y_test, y_predict)
print('loss : ', loss)
print('acc스코어 : ', acc_sc)
