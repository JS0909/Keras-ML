from sklearn.datasets import load_iris
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, LSTM
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from tensorflow.python.keras.callbacks import EarlyStopping
import tensorflow as tf

# 1. 데이터
datasets = load_iris()
print(datasets.DESCR)

print(datasets.feature_names)
x = datasets['data']
y = datasets['target']

#==========================================pandas.get_dummies===============================================================
y = pd.get_dummies(y)
print(y.shape)
print(y)
#===========================================================================================================================

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8,shuffle=True, random_state=9)
print(x_train.shape, x_test.shape)
x_train = x_train.reshape(120, 4, 1)
x_test = x_test.reshape(30, 4, 1)

# 2. 모델구성
model = Sequential()
model.add(LSTM(80, input_shape=(4,1), activation='relu'))
model.add(Dense(100))
model.add(Dropout(0.2))
model.add(Dense(90))
model.add(Dropout(0.1))
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

y_predict = model.predict(x_test)
y_predict = tf.argmax(y_predict, axis=1)
y_test = tf.argmax(y_test, axis=1)

acc_sc = accuracy_score(y_test, y_predict)
print('acc스코어 : ', acc_sc)

# DNN
# loss :  0.012709441594779491
# acc스코어 :  1.0

# LSTM
# loss :  0.009667815640568733
# acc스코어 :  1.0