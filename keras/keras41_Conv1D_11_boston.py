from sklearn.datasets import load_boston
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Conv1D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import time

# 1. 데이터
datasets = load_boston()
x, y = datasets.data, datasets.target

x_train, x_test, y_train, y_test =  train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(404, 13, 1)
x_test = x_test.reshape(102, 13, 1)

# 2. 모델구성 - dropout 적용 / 평가, 예측에는 안한다
model = Sequential()
model.add(Conv1D(100, 2, input_shape=(13, 1)))
model.add(Flatten())
model.add(Dropout(0.3)) # 위의 레이어에 드랍아웃 30프로를 적용 70프로만 연산 함/ 에포당 랜덤으로 빠짐
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dense(8))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam')
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50, restore_best_weights=True)
start_time=time.time()
log = model.fit(x_train, y_train, epochs=100, batch_size=32, callbacks=[es], validation_split=0.2, verbose=1)
end_time=time.time()

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss: ', loss)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2:', r2)
print('시간: ', end_time-start_time)

# DNN
# loss:  10.716378211975098
# r2: 0.8717874977761552

# LSTM
# loss:  25.709165573120117
# r2: 0.6924112938018632

# Conv1D
# loss:  10.66566276550293
# r2: 0.8723942600761079
# 시간:  8.552272319793701