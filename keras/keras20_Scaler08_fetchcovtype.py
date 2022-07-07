# keras18_gpu_test3 파일의 서머리를 확인해보시오.

from sklearn.datasets import fetch_covtype
from sklearn.preprocessing import OneHotEncoder
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import time

# 1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (581012, 54) (581012,)
print(np.unique(y, return_counts=True)) # [1 2 3 4 5 6 7]
# (array([1, 2, 3, 4, 5, 6, 7]), array([211840, 283301,  35754,   2747,   9493,  17367,  20510], dtype=int64))

#==========================================pandas.get_dummies===============================================================
y = pd.get_dummies(y)
print(y.shape)
print(y)
#===========================================================================================================================

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8,shuffle=True, random_state=9)
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler() 
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(np.min(x_train)) # 0.0
print(np.max(x_train)) # 1.0
print(np.min(x_test))
print(np.max(x_test))

#2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=54, activation='relu'))
model.add(Dense(200))
model.add(Dense(150))
model.add(Dense(300, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(7, activation='softmax'))

model.summary()

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
Es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50, restore_best_weights=True)
start_time = time.time()
log = model.fit(x_train, y_train, epochs=10, batch_size=50, callbacks=[Es], validation_split=0.2)
end_time = time.time()

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('accuracy : ', acc)

y_predict = model.predict(x_test)
y_predict = tf.argmax(y_predict, axis=1)
y_test = tf.argmax(y_test, axis=1)

acc_sc = accuracy_score(y_test, y_predict) # 비교
print('acc스코어 : ', acc_sc)
print('걸린 시간: ', end_time-start_time)

# 1. 스케일러 하기 전
# loss :  0.5608116388320923
# acc스코어 :  0.7621748147638185
# 걸린 시간:  110.55161762237549

# 2. MinMax scaler
# loss :  0.27308347821235657
# acc스코어 :  0.8890476149497001
# 걸린 시간:  103.8992919921875

# 3. Standard scaler
# loss :  0.2609665095806122
# acc스코어 :  0.8980577093534591
# 걸린 시간:  101.42629981040955

# 4. MaxAbsScaler
# loss :  0.276768296957016
# acc스코어 :  0.8883591645654587
# 걸린 시간:  104.17953681945801

# 5. RobustScaler
# loss :  0.24489960074424744
# acc스코어 :  0.9021023553608771
# 걸린 시간:  102.6204833984375