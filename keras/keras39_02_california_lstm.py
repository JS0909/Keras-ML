from sklearn.datasets import fetch_california_housing
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, LSTM
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler

# 1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test =  train_test_split(x, y, train_size=0.7, shuffle=True, random_state=66)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
print(x_train.shape, x_test.shape) # (14447, 8) (6193, 8)
x_train = x_train.reshape(14447, 8, 1)
x_test = x_test.reshape(6193, 8, 1)

# 2. 모델구성
# 시퀀셜 모델
model = Sequential()
model.add(LSTM(20, activation='relu', input_shape=(8, 1)))
model.add(Dense(20, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(10))
model.add(Dense(70))
model.add(Dropout(0.2))
model.add(Dense(50, activation='relu'))
model.add(Dense(10))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam', metrics=['mae'])
earlyStopping = EarlyStopping(monitor='val_loss', patience=30, mode='min', verbose=1, restore_best_weights=True)
hist = model.fit(x_train, y_train, epochs=200, batch_size=64, validation_split=0.2, callbacks=[earlyStopping], verbose=1)

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)

# loss :  [0.536990225315094, 0.5492812395095825]
# r2스코어 :  0.6086561961381309