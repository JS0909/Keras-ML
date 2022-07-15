import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.python.keras.callbacks import EarlyStopping

# 1 데이터
x1_datasets = np.array([range(100), range(301, 401)]) # 삼성전자 종가, 하이닉스 종가
x2_datasets = np.array([range(101, 201), range(411, 511), range(150, 250)]) # 원유, 돈육, 밀
x3_datasets = np.array([range(100, 200), range(1301, 1401)]) # 종이, 원달러환율

x1 = np.transpose(x1_datasets)
x2 = np.transpose(x2_datasets)
x3 = np.transpose(x3_datasets)

print(x1.shape, x2.shape, x3.shape) # (100, 2) (100, 3) (100, 2)

y1 = np.array(range(2001, 2101)) # 금리
y2 = np.array(range(201, 301)) # 환율

x1_train, x1_test, x2_train, x2_test, x3_train, x3_test, y1_train, y1_test, y2_train, y2_test = train_test_split(x1, x2, x3, y1, y2, 
                                                                                            train_size=0.7, shuffle=True, random_state=9)
print(x1_train.shape, x1_test.shape) # (70, 2) (30, 2)
print(x2_train.shape, x2_test.shape) # (70, 3) (30, 3)
print(x3_train.shape, x3_test.shape) # (70, 2) (30, 2)

# 2. 모델구성
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Input, Reshape, Conv1D, Conv2D, Flatten, GRU

# 2-1. 모델1
input1 = Input(shape=(2,))
dense1 = Dense(100, activation='relu', name='d1')(input1)
dense2 = Dense(100, activation='relu', name='d2')(dense1)
dense3 = Dense(100, activation='relu', name='d3')(dense2)
output1 = Dense(100, activation='relu', name='out_d1')(dense3)

# 2-2. 모델2
input2 = Input(shape=(3,))
dense11 = Dense(100, activation='relu')(input2)
re1 = Reshape(target_shape=(25, 2, 2))(dense11)
dense12 = Conv1D(100, 2, activation='swish', padding='same')(re1)
dense13 = Dense(100, activation='relu')(dense12)
dense14 = Conv2D(100, 2, activation='relu')(dense13)
re2 = Reshape(target_shape=(600, 4))(dense14)
dense15 = GRU(100, activation='relu')(re2)
output2 = Dense(100, activation='relu', name='out_d2')(dense15)
# 심심해서 섞어봄 근데 개 느리니까 돌리지 마라;

# 2-3. 모델3
input3 = Input(shape=(2,))
dense15 = Dense(100, activation='relu', name='d15')(input3)
dense16 = Dense(100, activation='swish', name='d16')(dense15)
dense17 = Dense(100, activation='relu', name='d17')(dense16)
dense18 = Dense(100, activation='relu', name='d18')(dense17)
output3 = Dense(100, activation='relu', name='out3')(dense18)

from tensorflow.python.keras.layers import concatenate, Concatenate
merge1 = concatenate([output1, output2, output3], name='m1')

# 2-4. y모델4
merge2 = Dense(100, activation='relu', name='mg2')(merge1)
merge3 = Dense(100, name='mg3')(merge2)
last_output1 = Dense(1, name='last1')(merge3)

# 2-5. y모델5
merge4 = Dense(100, activation='relu')(merge1)
merge5 = Dense(100)(merge4)
last_output2 = Dense(1, name='last2')(merge5)

model = Model(inputs=[input1, input2, input3], outputs=[last_output1, last_output2])

model.summary()

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
Es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50, restore_best_weights=True)
log = model.fit([x1_train, x2_train, x3_train], [y1_train, y2_train], epochs=10, batch_size=1, callbacks=[Es], validation_split=0.25)

# 4. 평가, 예측
loss1 = model.evaluate([x1_test, x2_test, x3_test], y1_test)
loss2 = model.evaluate([x1_test, x2_test, x3_test], y2_test)
print('loss1: ', loss1)
print('loss2: ', loss2)
y_predict1, y_predict2 = model.predict([x1_test, x2_test, x3_test])
print(y_predict1.shape)
r2_1 = r2_score(y1_test, y_predict1)
r2_2 = r2_score(y2_test, y_predict2)
print('r2_1: ', r2_1)
print('r2_2: ', r2_2)
print('ensemble3')

# y모델 분화 안한 것
# loss1:  [3239456.0, 0.012645941227674484, 3239456.0, 0.10090331733226776, 1799.8489990234375]
# loss2:  [3240120.0, 3240120.0, 0.0305319856852293, 1800.0333251953125, 0.151081845164299]
# r2_1:  0.9999797360895613
# r2_2:  0.9999510754116319

# y모델 분화
# loss1:  [3240038.25, 0.0087300855666399, 3240038.25, 0.08511149138212204, 1800.0106201171875]
# loss2:  [3239699.25, 3239699.25, 0.00020341604249551892, 1799.9163818359375, 0.01277058944106102]
# r2_1:  0.9999860108737827
# r2_2:  0.9999996740452488