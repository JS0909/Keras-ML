from sklearn.datasets import load_wine
from sklearn.preprocessing import OneHotEncoder
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from tensorflow.python.keras.callbacks import EarlyStopping
import time

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
model.add(Dense(80, input_dim=13, activation='relu'))
model.add(Dense(100))
model.add(Dense(90))
model.add(Dense(70, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(3, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
Es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50, restore_best_weights=True)
start_time = time.time()
log = model.fit(x_train, y_train, epochs=1000, batch_size=100, callbacks=[Es], validation_split=0.2)
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

# 1. 스케일러 하기 전
# loss :  0.07546340674161911
# acc스코어 :  0.9722222222222222
# 걸린 시간:  7.713343381881714

# 2. MinMax scaler
# loss :  0.0013649107422679663
# acc스코어 :  1.0
# 걸린 시간:  5.220351696014404

# 3. Standard scaler
# loss :  0.015283097513020039
# acc스코어 :  1.0
# 걸린 시간:  19.361830711364746

# 4. MaxAbsScaler
# loss :  0.011181176640093327
# acc스코어 :  1.0
# 걸린 시간:  3.5697598457336426

# 5. RobustScaler
# loss :  0.043323814868927
# acc스코어 :  0.9722222222222222
# 걸린 시간:  2.253476142883301