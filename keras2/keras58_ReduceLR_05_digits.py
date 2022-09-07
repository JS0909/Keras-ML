from sklearn.datasets import load_digits
from sklearn.preprocessing import OneHotEncoder
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# 1. 데이터
datasets = load_digits()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (1797, 64) (1797,)
# 이미지 파일 8x8의 픽셀 한칸짜리가 1797개가 있음, 각 픽셀에 칠해진 여부에 때라 0 ~ 255의 숫자로 표시
print(np.unique(y, return_counts=True)) # [0 1 2 3 4 5 6 7 8 9] <-이 숫자를 찾아라
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), array([178, 182, 177, 183, 181, 182, 181, 179, 174, 180], dtype=int64))


#==========================================pandas.get_dummies===============================================================
y = pd.get_dummies(y)
print(y.shape)
print(y)
#===========================================================================================================================

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8,shuffle=True, random_state=9)

                          
#2. 모델구성
model = Sequential()
model.add(Dense(80, input_dim=64, activation='relu'))
model.add(Dense(100))
model.add(Dense(90))
model.add(Dense(70, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

Es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto', verbose=1, factor=0.5)

log = model.fit(x_train, y_train, epochs=1000, batch_size=100, callbacks=[Es, reduce_lr], validation_split=0.2)

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('accuracy : ', acc)


y_predict = model.predict(x_test)
y_predict = tf.argmax(y_predict, axis=1)
print(y_test)
y_test = tf.argmax(y_test, axis=1)

print(y_test)
print(y_predict)

acc_sc = accuracy_score(y_test, y_predict)
print('acc스코어 : ', acc_sc)

# loss :  0.10494104772806168
# accuracy :  0.9722222089767456
# acc스코어 :  0.9722222222222222

# 갱신 없음
# acc스코어 :  0.9611111111111111