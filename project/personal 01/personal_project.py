from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Input
import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping
import tensorflow as tf
from sklearn.metrics import accuracy_score
import pandas as pd
from tensorflow.python.keras.layers import concatenate, Concatenate

tf.random.set_seed(9) # 하이퍼 파라미터 튜닝 용이하게 하기 위해

# 1. 데이터
filepath = 'd:/study_data/_save/_npy/_project/'
suffix = '.npy'

x_train = np.load(filepath+'train_x'+suffix)
y1_train = np.load(filepath+'train_y1'+suffix)
y2_train = np.load(filepath+'train_y2'+suffix)

x_test = np.load(filepath+'test_x'+suffix)
y1_test = np.load(filepath+'test_y1'+suffix)
y2_test = np.load(filepath+'test_y2'+suffix)

testing_img = np.load(filepath+'testing_img'+suffix)

# print(x_train.shape) # (4000, 150, 150, 3)
# print(y1_train.shape, y2_train.shape) # (6438, 30) (6438, 4)
# print(y1_test.shape, y2_test.shape) # (1610, 30) (1610, 4)

# print(y2_train)
# 2. 모델구성
# 2-1. input모델
input1 = Input(shape=(150, 150, 3))
conv1 = Conv2D(32,(2,2), padding='same', activation='swish')(input1)
mp1 = MaxPool2D()(conv1)
conv2 = Conv2D(32,(2,2), activation='swish')(mp1)
flat1 = Flatten()(conv2)
dense1 = Dense(32, activation='relu')(flat1)
dense2 = Dense(32, activation='relu')(dense1)
output = Dense(32, activation='relu')(dense2)

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
log = model.fit(x_train, [y1_train, y2_train], epochs=10, batch_size=32, callbacks=[Es], validation_split=0.2)
model.save('D:/study_data/_save/_h5/project.h5')

# model = load_model('D:/study_data/_save/_h5/project.h5')

#4. 평가, 예측
loss = model.evaluate(x_test, [y1_test, y2_test])
print('tested loss : ', loss)

y1_pred, y2_pred = model.predict(x_test)
y1_pred = tf.argmax(y1_pred, axis=1)
y1_test_arg = tf.argmax(y1_test, axis=1)
y2_pred = tf.argmax(y2_pred, axis=1)
y2_test_arg = tf.argmax(y2_test, axis=1)
acc_sc1 = accuracy_score(y1_test_arg,y1_pred)
acc_sc2 = accuracy_score(y2_test_arg,y2_pred)
print('y1_acc스코어 : ', acc_sc1)
print('y2_acc스코어 : ', acc_sc2)

y1_pred = np.array(y1_pred)
y2_pred = np.array(y2_pred)
a = range(0, 10)
for i in a:
    print(y1_pred[i])
    print(y2_pred[i], '\n')

testpred_breed, testpred_age = model.predict(testing_img)
testpred_breed = tf.argmax(testpred_breed, axis=1)
testpred_age = tf.argmax(testpred_age, axis=1)
testpred_breed = np.array(testpred_breed)
testpred_age = np.array(testpred_age)
print(testpred_breed, testpred_age)

converter = tf.lite.TFLiteConverter.from_keras_model(model) 
tflite_model = converter.convert() 
open("d:/study_data/_save/converted_model.tflite", "wb").write(tflite_model)
