from matplotlib.cbook import flatten
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Dropout, Conv1D, Flatten
from tensorflow.keras.datasets import fashion_mnist
import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping
import tensorflow as tf
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical

# 1. 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# import matplotlib.pyplot as plt
# plt.imshow(x_train[0], 'gray') # 이미지 보여주기
# plt.show()

print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,)

print(np.unique(y_train, return_counts=True)) # (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000],
      # dtype=int64))


y_train= to_categorical(y_train)
y_test=to_categorical(y_test)

# 2. 모델구성
model = Sequential()
model.add(Conv1D(80, 2, input_shape=(28,28)))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.3))
model.add(Dense(90, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(70))
model.add(Dropout(0.1))
model.add(Dense(50))
model.add(Dense(10, activation='softmax'))

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
Es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50, restore_best_weights=True)
log = model.fit(x_train, y_train, epochs=10, batch_size=32, callbacks=[Es], validation_split=0.2)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
y_predict = tf.argmax(y_predict, axis=1)
y_test = tf.argmax(y_test, axis=1)
acc_sc = accuracy_score(y_test, y_predict)
print('loss : ', loss)
print('acc스코어 : ', acc_sc)
print('fashionmnist')

# DNN
# loss :  [0.4175029397010803, 0.8562999963760376]
# acc스코어 :  0.8563

# 함수형
# loss :  [0.4193294048309326, 0.8574000000953674]
# acc스코어 :  0.8574

# LSTM
# loss :  [0.4825081527233124, 0.8259000182151794]
# acc스코어 :  0.8259

# Conv1D
# loss :  [1.0244991779327393, 0.5584999918937683]
# acc스코어 :  0.5585
