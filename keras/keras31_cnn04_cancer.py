import numpy as np
from sklearn.datasets import load_breast_cancer
from tensorflow.python.keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from tensorflow.python.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

# 1. 데이터
datasets = load_breast_cancer()

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test =  train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

# Scaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape, x_test.shape) # (569, 30) (569,)

x_train = x_train.reshape(455, 5, 6, 1)
x_test = x_test.reshape(114, 5, 6, 1)

# 2. 모델구성
# 시퀀셜
model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(2,2), strides=1, padding='same', input_shape=(5, 6, 1)))
model.add(MaxPool2D((2,2), padding='same'))
model.add(Conv2D(10, (1,1),padding='valid', activation='relu'))
model.add(Conv2D(5, (1,1),padding='same', activation='relu'))
model.add(Conv2D(4, (1,1),padding='valid', activation='relu'))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 3. 컴파일, 훈련
model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['accuracy', 'mse']) 
earlyStopping = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1, restore_best_weights=True)
hist = model.fit(x_train, y_train, epochs=1000, batch_size=50, callbacks=[earlyStopping], validation_split=0.25)


# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)
y_predict = np.round(y_predict,0)
acc = accuracy_score(y_test, y_predict)
print('acc스코어 : ', acc)

# DNN
# loss :  [0.2820783853530884, 0.8859649300575256, 0.08301880955696106]
# acc스코어 :  0.8859649122807017

# CNN
# loss :  [0.1587904393672943, 0.9385964870452881, 0.036773715168237686]
# acc스코어 :  0.9385964912280702