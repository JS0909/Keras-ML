import numpy as np
from sklearn.datasets import load_breast_cancer
from tensorflow.python.keras.layers import Dense, Flatten, Conv1D
from tensorflow.python.keras.models import Sequential
from sklearn.model_selection import train_test_split
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 1. 데이터
datasets = load_breast_cancer()

x = datasets.data # datasets['data']
y = datasets.target # datasets['target'] // key value니까 이렇게도 가능
print(x.shape, y.shape) # (569, 30) (569,)

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
model.add(Conv1D(5, 2, activation='relu', input_shape=(30, 1)))
model.add(Flatten())
model.add(Dense(50, activation=None)) # activation defualt = None (linear)
model.add(Dense(40, activation = 'relu')) # !중간에서만 쓸 수 있다, 평타 85% 이상 개좋은 놈
model.add(Dense(100, activation='sigmoid'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid')) # 0과 1 사이의 유리수로 최종 out put이 저장된다

# 3. 컴파일, 훈련
model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['accuracy', 'mse']) 

from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1, restore_best_weights=True)
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size=50,
                callbacks=[earlyStopping],
                validation_split=0.25)
end_time = time.time()

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)
from sklearn.metrics import r2_score, accuracy_score
y_predict = np.round(y_predict,0)
acc = accuracy_score(y_test, y_predict)
print('acc스코어 : ', acc)
print('걸린 시간: ', end_time-start_time)


# Standard scaler
# loss :  [0.07493800669908524, 0.9736841917037964, 0.020953446626663208]
# acc스코어 :  0.9736842105263158
# 걸린 시간:  2.553802967071533

# LSTM
# loss :  [0.08698250353336334, 0.9824561476707458, 0.021481234580278397]
# acc스코어 :  0.9824561403508771
# 걸린 시간:  35.46809959411621

# Conv1D
# loss :  [0.11493552476167679, 0.9561403393745422, 0.027350271120667458]
# acc스코어 :  0.956140350877193
# 걸린 시간:  7.407166481018066