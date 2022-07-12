from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from tensorflow.python.keras.callbacks import EarlyStopping
import tensorflow as tf

tf.random.set_seed(99)
# y = wx + b의 w값을 처음 랜덤으로 긋는 것을 어떻게 그을지 고정하고 시작

# 1. 데이터
datasets = load_iris()
print(datasets.DESCR) # 상관관계가 -라면 뺄 수 있다 근데 진짜 상관관계가 -0.4인지는 확인 해봐야함
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

#==========================================to_categorical==================================================================
# from tensorflow.keras.utils import to_categorical
# y = to_categorical(y) # 인코딩 방식때문에 쉐이프가 1개 더 생겼음, 그래서 OneHotEncoder나 get_dummies를 사용해줘야함, 차이는 keras16 파일에서
# print(y.shape) #(581012, 8))
# print(y)
#===========================================================================================================================

#==========================================OneHotEncoder====================================================================
from sklearn.preprocessing import OneHotEncoder
oh = OneHotEncoder()
print(y.shape) # (581012,)
y = datasets.target.reshape(-1,1) # reshape 전은 벡터로, reshape 후에 행렬로
print(y.shape) # (581012, 1)
oh.fit(y)
y = oh.transform(y).toarray()
print(y)
print(y.shape)
#===========================================================================================================================

#==========================================pandas.get_dummies===============================================================
y = pd.get_dummies(y)
print(y.shape)
print(y)
#===========================================================================================================================
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8,shuffle=True, random_state=9)
# 안섞으면 2가 trainset에 안나옴 순차적으로 0부터 나오는 걸 짜르게 되니까
# print(y_train)
# print(y_test)
                          
#2. 모델구성
model = Sequential()
model.add(Dense(80, input_dim=4, activation='relu'))
model.add(Dense(100))
model.add(Dense(90))
model.add(Dense(70, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(3, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
Es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50, restore_best_weights=True)
log = model.fit(x_train, y_train, epochs=100, batch_size=100, callbacks=[Es], validation_split=0.2)

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
y_predict = tf.argmax(y_predict, axis=1) # 행끼리 비교해서 몇번째 인덱스가 제일 큰지 알려줌
y_test = tf.argmax(y_test, axis=1) # y_test도 argmax를 해서 같은 리스트를 비교하기

print(y_test)
print(y_predict)

acc_sc = accuracy_score(y_test, y_predict) # 비교
print('acc스코어 : ', acc_sc)

# error : 문제가 있어서 중단됨
# bug : 작동은 되지만 문제가 있음

# loss :  0.004345672205090523
# accuracy :  1.0