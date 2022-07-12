from sklearn.datasets import fetch_california_housing
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import datetime
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

# 1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test =  train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)
print(x_train.shape, x_test.shape)

# Scaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(16512, 4, 2, 1) # 데이터의 갯수자체는 성능과 큰 상관이 없을 수 있다
x_test = x_test.reshape(4128, 4, 2, 1)

# 2. 모델구성
model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(2,2), strides=1, padding='same', input_shape=(4, 2, 1)))
model.add(MaxPool2D((2,2), padding='same'))
model.add(Conv2D(10, (1,1),padding='valid', activation='relu'))
model.add(Conv2D(5, (1,1),padding='same', activation='relu'))
model.add(Conv2D(4, (1,1),padding='valid', activation='relu'))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam', metrics=['mae'])
earlyStopping = EarlyStopping(monitor='val_loss', patience=30, mode='min', verbose=1, restore_best_weights=True)
hist = model.fit(x_train, y_train, epochs=200, batch_size=50,
                callbacks=[earlyStopping],
                validation_split=0.25)

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)

# loss :  [0.5216111540794373, 0.546445369720459]
# r2스코어 :  0.625834467060862