from sklearn.datasets import load_wine
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, LSTM
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# 1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (178, 13) (178,)
print(np.unique(y, return_counts=True))

#==========================================pandas.get_dummies===============================================================
y = pd.get_dummies(y)
print(y.shape)
print(y)
#===========================================================================================================================

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8,shuffle=True, random_state=9)
print(x_train.shape, x_test.shape)
x_train = x_train.reshape(142, 13, 1)
x_test = x_test.reshape(36, 13, 1)
        
#2. 모델구성
model = Sequential()
model.add(LSTM(80, input_shape=(13, 1), activation='relu'))
model.add(Dense(100))
model.add(Dropout(0.4))
model.add(Dense(90))
model.add(Dense(70, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(50, activation='relu'))
model.add(Dense(3, activation='softmax'))


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
y_test = tf.argmax(y_test, axis=1)

acc_sc = accuracy_score(y_test, y_predict)
print('acc스코어 : ', acc_sc)


# LSTM
# loss :  0.00020670965022873133
# accuracy :  1.0
# acc스코어 :  1.0

# 갱신 안됨
# loss :  0.3932833969593048
# accuracy :  0.8055555820465088
# acc스코어 :  0.8055555555555556