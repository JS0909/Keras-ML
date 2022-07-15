import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.python.keras.callbacks import EarlyStopping

# 1 데이터
x1_datasets = np.array([range(100), range(301, 401)]) # 삼성전자 종가, 하이닉스 종가
x2_datasets = np.array([range(101, 201), range(411, 511), range(150, 250)]) # 원유, 돈육, 밀
x1 = np.transpose(x1_datasets)
x2 = np.transpose(x2_datasets)

print(x1.shape, x2.shape) # (100, 2) (100, 3)

y = np.array(range(2001, 2101)) # 금리

x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1, x2, y, train_size=0.7, shuffle=True, random_state=9)
print(x1_train.shape, x1_test.shape) # (70, 2) (30, 2)
print(x2_train.shape, x2_test.shape) # (70, 3) (30, 3)
print(y_train.shape, y_test.shape)   # (70,) (30,)

# 2. 모델구성
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Input

# 2-1. 모델1
input1 = Input(shape=(2,))
dense1 = Dense(100, activation='relu', name='d1')(input1)
dense2 = Dense(100, activation='relu', name='d2')(dense1)
dense3 = Dense(100, activation='relu', name='d3')(dense2)
output1 = Dense(100, activation='relu', name='out_d1')(dense3)

# 2-2. 모델2
input2 = Input(shape=(3,))
dense11 = Dense(100, activation='relu', name='d11')(input2)
dense12 = Dense(100, activation='swish', name='d12')(dense11)
dense13 = Dense(100, activation='relu', name='d13')(dense12)
dense14 = Dense(100, activation='relu', name='d14')(dense13)
output2 = Dense(100, activation='relu', name='out_d2')(dense14)

from tensorflow.python.keras.layers import concatenate, Concatenate
merge1 = concatenate([output1, output2], name='m1')
merge2 = Dense(100, activation='relu', name='mg2')(merge1)
merge3 = Dense(100, name='mg3')(merge2)
last_output = Dense(1, name='last')(merge3)

model = Model(inputs=[input1, input2], outputs=[last_output])

model.summary()

# Model: "model"
# __________________________________________________________________________________________________
# Layer (type)                    Output Shape         Param #     Connected to
# ==================================================================================================
# input_2 (InputLayer)            [(None, 3)]          0
# __________________________________________________________________________________________________
# input_1 (InputLayer)            [(None, 2)]          0
# __________________________________________________________________________________________________
# d11 (Dense)                     (None, 11)           44          input_2[0][0]
# __________________________________________________________________________________________________
# d1 (Dense)                      (None, 1)            3           input_1[0][0]
# __________________________________________________________________________________________________
# d12 (Dense)                     (None, 12)           144         d11[0][0]
# __________________________________________________________________________________________________
# d2 (Dense)                      (None, 2)            4           d1[0][0]
# __________________________________________________________________________________________________
# d13 (Dense)                     (None, 13)           169         d12[0][0]
# __________________________________________________________________________________________________
# d3 (Dense)                      (None, 3)            9           d2[0][0]
# __________________________________________________________________________________________________
# d14 (Dense)                     (None, 14)           196         d13[0][0]
# __________________________________________________________________________________________________
# out_d1 (Dense)                  (None, 10)           40          d3[0][0]
# __________________________________________________________________________________________________
# out_d2 (Dense)                  (None, 10)           150         d14[0][0]
# __________________________________________________________________________________________________
# m1 (Concatenate)                (None, 20)           0           out_d1[0][0]
#                                                                  out_d2[0][0]
# __________________________________________________________________________________________________
# mg2 (Dense)                     (None, 2)            42          m1[0][0]
# __________________________________________________________________________________________________
# mg3 (Dense)                     (None, 3)            9           mg2[0][0]
# __________________________________________________________________________________________________
# last (Dense)                    (None, 1)            4           mg3[0][0]
# ==================================================================================================
# Total params: 814
# Trainable params: 814
# Non-trainable params: 0
# __________________________________________________________________________________________________

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
Es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50, restore_best_weights=True)
log = model.fit([x1_train, x2_train], y_train, epochs=100, batch_size=1, callbacks=[Es], validation_split=0.25)

# 4. 평가, 예측
loss = model.evaluate([x1_test, x2_test], y_test)
print('loss: ', loss)
print('삼성전자,하이닉스 종가: ', x1_test)
print('원유, 돈육, 밀: ', x2_test)
y_predict = model.predict([x1_test, x2_test])
r2 = r2_score(y_test, y_predict)
print('프레딕트: ', y_predict)
print('r2: ', r2)
print('ensemble1')

# loss:  [0.008920214138925076, 0.09355875849723816]
# r2:  0.9999857062088481