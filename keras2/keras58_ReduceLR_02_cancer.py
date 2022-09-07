from sklearn.datasets import load_breast_cancer
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM, Dense, Dropout, Conv1D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import time
from tensorflow.python.keras.optimizer_v2 import adam

# 1. 데이터
datasets = load_breast_cancer()

x = datasets.data 
y = datasets.target 

x_train, x_test, y_train, y_test =  train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape, x_test.shape)
x_train = x_train.reshape(455, 30, 1)
x_test = x_test.reshape(114, 30, 1)


# 2. 모델구성
model = Sequential()
model.add(LSTM(5, activation='relu', input_shape=(30, 1)))
model.add(Dense(50, activation=None))
model.add(Dense(40, activation = 'relu')) 
model.add(Dense(100, activation='sigmoid'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid')) 


# 3. compilel, fit
optimizer = adam.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, metrics=['acc'], loss='binary_crossentropy')

es = EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto', verbose=1, factor=0.5) # learning rate를 0.5만큼 감축시키겠다

start = time.time()
model.fit(x_train, y_train, epochs=300, validation_split=0.2, batch_size=128, callbacks=[es,reduce_lr])
end = time.time()-start

loss, acc = model.evaluate(x_test,y_test)

print('걸린 시간: ', end)
print('loss: ', loss)
print('acc: ', acc)

# 걸린 시간:  107.07294178009033
# loss:  0.12232352793216705
# acc:  0.9561403393745422
# 갱신 없음