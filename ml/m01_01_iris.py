from sklearn.datasets import load_iris
# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from tensorflow.python.keras.callbacks import EarlyStopping
import tensorflow as tf

from sklearn.svm import LinearSVC # 모델
# 서포트 벡터 머신 / 리니어 서포트 벡터 클레시파이어
# 원핫 x, 컴파일 x, argmax x

tf.random.set_seed(99)
# y = wx + b의 w값을 처음 랜덤으로 긋는 것을 어떻게 그을지 고정하고 시작

# 1. 데이터
datasets = load_iris()
print(datasets.DESCR)
'''
- class:
                - Iris-Setosa
                - Iris-Versicolour
                - Iris-Virginica
y값이 3개
이 3개 꽃 중 하나가 나와야 함
3중 분류
'''
print(datasets.feature_names)
x = datasets['data']
y = datasets['target']
print(x, '\n', y)
print(x.shape) # (150, 4)
print(y.shape) # (150,)
print('y의 라벨값: ', np.unique(y))

# sklearn에서는 인코딩 필요 없음
# y = pd.get_dummies(y)
# print(y.shape)
# print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8,shuffle=True, random_state=9)
                      
#2. 모델구성
# model = Sequential()
# model.add(Dense(80, input_dim=4, activation='relu'))
# model.add(Dense(100))
# model.add(Dense(90))
# model.add(Dense(70, activation='relu'))
# model.add(Dense(50, activation='relu'))
# model.add(Dense(3, activation='softmax'))

model = LinearSVC()


#3. 컴파일, 훈련
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# sklearn에서는 컴파일 없음
# Es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50, restore_best_weights=True)
# log = model.fit(x_train, y_train, epochs=100, batch_size=100, callbacks=[Es], validation_split=0.2)

model.fit(x_train, y_train) # 컴파일도 포함되어 있음

#4. 평가, 예측
# loss, acc = model.evaluate(x_test, y_test)
# print('loss : ', loss)
# print('accuracy : ', acc)

# print(y_test)
# print(y_predict)

result = model.score(x_test, y_test) # evaluate 대신 score 사용
print('acc 결과: ', result) # 분류모델에서는 acc score // 회귀모델에서는 R2 score 자동으로 나옴

y_predict = model.predict(x_test)
# y_predict = tf.argmax(y_predict, axis=1)
# y_test = tf.argmax(y_test, axis=1)

acc_sc = accuracy_score(y_test, y_predict)
print('acc스코어 : ', acc_sc)

# loss :  0.005208590067923069
# accuracy :  1.0
# tf.Tensor([2 1 2 2 1 0 0 0 1 0 0 1 1 1 0 1 0 1 2 0 0 0 2 0 2 1 0 2 0 2], shape=(30,), dtype=int64)
# tf.Tensor([2 1 2 2 1 0 0 0 1 0 0 1 1 1 0 1 0 1 2 0 0 0 2 0 2 1 0 2 0 2], shape=(30,), dtype=int64)
# acc스코어 :  1.0

# 머신러닝 LinearSVC
# 결과:  1.0
# acc스코어 :  1.0
# 아주 빠름, 단층레이어임