from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.python.keras.callbacks import EarlyStopping
import tensorflow as tf
import time

tf.random.set_seed(99)

# 1. 데이터
datasets = load_iris()
print(datasets.DESCR)

print(datasets.feature_names)
x = datasets['data']
y = datasets['target']
print(x, '\n', y)
print(x.shape) # (150, 4)
print(y.shape) # (150,)
print('y의 라벨값: ', np.unique(y))

#==========================================pandas.get_dummies===============================================================
y = pd.get_dummies(y)
print(y.shape)
print(y)
#===========================================================================================================================

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8,shuffle=True, random_state=9)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler() 
# scaler = RobustScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)
# print(np.min(x_train))
# print(np.max(x_train))
# print(np.min(x_test))
# print(np.max(x_test))

# 2. 모델구성
# 시퀀셜
# model = Sequential()
# model.add(Dense(80, input_dim=4, activation='relu'))
# model.add(Dense(100))
# model.add(Dense(90))
# model.add(Dense(70, activation='relu'))
# model.add(Dense(50, activation='relu'))
# model.add(Dense(3, activation='softmax'))

# 함수
input1 = Input(shape=(4,))
dense1 = Dense(80, activation='relu')(input1)
dense2 = Dense(100)(dense1)
dense3 = Dense(90)(dense2)
dense4 = Dense(70, activation='relu')(dense3)
dense5 = Dense(50, activation='relu')(dense4)
output1 = Dense(3, activation='softmax')(dense5)
model = Model(inputs=input1,outputs=output1)

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
Es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50, restore_best_weights=True)
start_time = time.time()
log = model.fit(x_train, y_train, epochs=100, batch_size=100, callbacks=[Es], validation_split=0.2)
end_time = time.time()

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('accuracy : ', acc)

y_predict = model.predict(x_test)
y_predict = tf.argmax(y_predict, axis=1)
y_test = tf.argmax(y_test, axis=1)

acc_sc = accuracy_score(y_test, y_predict)
print('acc스코어 : ', acc_sc)
print('걸린 시간: ', end_time-start_time)

# 시퀀셜 without scaler
# loss :  0.005203894339501858
# accuracy :  1.0
# 걸린 시간:  2.3607850074768066

# 함수 wihout scaler
# accuracy :  1.0
# acc스코어 :  1.0
# 걸린 시간:  2.3672475814819336