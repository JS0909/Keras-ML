from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.python.keras.optimizer_v2 import adam
import time

# 1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test =  train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape, x_test.shape)

# 2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=13))
model.add(Dense(10))
model.add(Dense(30, activation='relu'))
model.add(Dense(40))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))

# 3. compilel, fit
optimizer = adam.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, metrics=['mae'], loss='mse')

es = EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto', verbose=1, factor=0.5)

start = time.time()
model.fit(x_train, y_train, epochs=300, validation_split=0.2, batch_size=128, callbacks=[es,reduce_lr])
end = time.time()-start

loss, mae = model.evaluate(x_test,y_test)
y_pred = model.predict(x_test)

print('걸린 시간: ', end)
print('loss: ', loss)
print('r2: ', r2_score(y_test, y_pred))

# 0.7266701456479958

# 걸린 시간:  6.66641092300415
# loss:  15.149988174438477
# r2:  0.8187430638264707