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
import time

# 1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (581012, 54) (581012,)
print(np.unique(y, return_counts=True)) # [1 2 3 4 5 6 7]
# (array([1, 2, 3, 4, 5, 6, 7]), array([211840, 283301,  35754,   2747,   9493,  17367,  20510], dtype=int64))

#==========================================to_categorical==================================================================
# from tensorflow.keras.utils import to_categorical
# y = to_categorical(y) # 인코딩 방식때문에 쉐이프가 1개 더 생겼음, 그래서 OneHotEncoder나 get_dummies를 사용해줘야함, 차이는 keras16 파일에서
# print(y.shape) #(581012, 8))
# print(y)
#===========================================================================================================================

#==========================================OneHotEncoder====================================================================
# from sklearn.preprocessing import OneHotEncoder
# oh = OneHotEncoder()
# print(y.shape) # (581012,)
# y = datasets.target.reshape(-1,1) # reshape 전은 벡터로, reshape 후에 행렬로
# print(y.shape) # (581012, 1)
# oh.fit(y)
# y = oh.transform(y).toarray()
# print(y)
# print(y.shape)
#===========================================================================================================================

#==========================================pandas.get_dummies===============================================================
y = pd.get_dummies(y)
print(y.shape)
print(y)
#===========================================================================================================================
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8,shuffle=True, random_state=9)

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
# batch_size 디폴트는 보통 32로 한다 32, 64 ~~ 순으로 올려봄 보통

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('accuracy : ', acc)

# 위에꺼랑 똑같이 프린트됨
# results = model.evaluate(x_test, y_test)
# print('loss : ', results[0])
# print('accuracy : ', results[1])
'''
print('============ y_test[:5]================')
print(y_test[:5])
print('============ y_pred[:5]================')
y_pred = model.predict(x_test[:5]) # predict 값은 9.0022 ~이런식으로 나오니까 세개중에 제일 큰거만 1로 만들어버려야 비교 가능
print(y_pred)
'''

y_predict = model.predict(x_test)
y_predict = tf.argmax(y_predict, axis=1)
y_test = tf.argmax(y_test, axis=1)

print(y_test)
print(y_predict)

acc_sc = accuracy_score(y_test, y_predict) # 비교
print('acc스코어 : ', acc_sc)

print('걸린 시간: ', end_time-start_time)

# GPU 사용시 걸린 시간:  317.5111241340637
# CPU 사용시 걸린 시간:  65.95031785964966

# GPU 걸린 시간:  327.42542028427124
# CPU 걸린 시간:  101.80242276191711

# Total params: 162,757
# GPU 걸린 시간:  321.1235160827637