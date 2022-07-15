import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.python.keras.callbacks import EarlyStopping

# 1 데이터
x1_datasets = np.array([range(100), range(301, 401)]) # 삼성전자 종가, 하이닉스 종가
x1 = np.transpose(x1_datasets)


print(x1.shape) # (100, 2) (100, 3) (100, 2)

y1 = np.array(range(2001, 2101)) # 금리
y2 = np.array(range(201, 301)) # 환율

x1_train, x1_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
    x1, y1, y2, train_size=0.7, shuffle=True, random_state=9)
    
print(x1_train.shape, x1_test.shape) # (70, 2) (30, 2)

# 2. 모델구성
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Input, Reshape, Conv1D, Conv2D, Flatten, GRU

# 2-1. 모델1
input1 = Input(shape=(2,))
dense1 = Dense(100, activation='relu', name='d1')(input1)
dense2 = Dense(100, activation='relu', name='d2')(dense1)
dense3 = Dense(100, activation='relu', name='d3')(dense2)
output1 = Dense(100, activation='relu', name='out_d1')(dense3)

# 2-4. output모델1
output41 = Dense(10)(output1)
output42 = Dense(10)(output41)
last_output1 = Dense(1)(output42)

# 2-5. output모델2
output51 = Dense(10)(output1)
output52 = Dense(10)(output51)
last_output2 = Dense(1)(output52)

model = Model(inputs=[input1], outputs=[last_output1, last_output2])

model.summary()

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
Es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50, restore_best_weights=True)
log = model.fit([x1_train], [y1_train, y2_train], epochs=100, batch_size=1, callbacks=[Es], validation_split=0.25)

# 4. 평가, 예측
loss1 = model.evaluate([x1_test], y1_test)
loss2 = model.evaluate([x1_test], y2_test)
print('loss1: ', loss1)
print('loss2: ', loss2)
y_predict = model.predict([x1_test])
r2 = r2_score(y1_test, y_predict[0])
r2_2 = r2_score(y2_test, y_predict[1])
print('r2: ', r2, r2_2)
print('ensemble4')

# loss1:  [합친거: 3239965.25, 0.03318199887871742, 3239965.25]
# loss2:  [합친거: 3239916.0, 3239916.0, 0.00041347762453369796]
# r2:  0.9999468290228889 0.9999993374416222