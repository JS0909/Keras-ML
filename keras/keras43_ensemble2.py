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

y = np.array(range(2001, 2101)) # 금리

x1_train, x1_test, x2_train, x2_test, x3_train, x3_test, y_train, y_test = train_test_split(x1, x2, x3, y, 
                                                                                            train_size=0.7, shuffle=True, random_state=9)
print(x1_train.shape, x1_test.shape) # (70, 2) (30, 2)
print(x2_train.shape, x2_test.shape) # (70, 3) (30, 3)
print(x3_train.shape, x3_test.shape) # (70, 2) (30, 2)
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

# 2-3. 모델3
input3 = Input(shape=(2,))
dense15 = Dense(100, activation='relu', name='d15')(input3)
dense16 = Dense(100, activation='swish', name='d16')(dense15)
dense17 = Dense(100, activation='relu', name='d17')(dense16)
dense18 = Dense(100, activation='relu', name='d18')(dense17)
output3 = Dense(100, activation='relu', name='out3')(dense18)

from tensorflow.python.keras.layers import concatenate, Concatenate
merge1 = concatenate([output1, output2, output3], name='m1')
merge2 = Dense(100, activation='relu', name='mg2')(merge1)
merge3 = Dense(100, name='mg3')(merge2)
last_output = Dense(1, name='last')(merge3)

model = Model(inputs=[input1, input2, input3], outputs=[last_output])

model.summary()

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
Es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50, restore_best_weights=True)
log = model.fit([x1_train, x2_train, x3_train], y_train, epochs=100, batch_size=1, callbacks=[Es], validation_split=0.25)

# 4. 평가, 예측
loss = model.evaluate([x1_test, x2_test, x3_test], y_test)
print('loss: ', loss)
y_predict = model.predict([x1_test, x2_test, x3_test])
r2 = r2_score(y_test, y_predict)
print('프레딕트: ', y_predict)
print('r2: ', r2)
print('삼성전자,하이닉스 종가: ', x1_test)
print('원유, 돈육, 밀: ', x2_test)
print('종이, 원달러환율: ', x3_test)
print('ensemble2')

# loss:  [0.08607950061559677, 0.2611490786075592]
# r2:  0.9998620658261187
# 삼성전자,하이닉스 종가:  
# [[ 75 376]
#  [ 42 343]
#  [ 46 347]
#  [ 68 369]
#  [  3 304]
#  [ 39 340]
#  [ 23 324]
#  [ 20 321]
#  [ 70 371]
#  [ 73 374]
#  [ 41 342]
#  [ 26 327]
#  [ 32 333]
#  [ 25 326]
#  [ 95 396]
#  [ 83 384]
#  [  6 307]
#  [ 44 345]
#  [ 21 322]
#  [ 28 329]
#  [ 82 383]
#  [ 31 332]
#  [ 48 349]
#  [ 78 379]
#  [ 14 315]
#  [ 36 337]
#  [ 51 352]
#  [ 86 387]
#  [ 61 362]
#  [ 55 356]]
# 원유, 돈육, 밀:  
# [[176 486 225]
#  [143 453 192]
#  [147 457 196]
#  [169 479 218]
#  [104 414 153]
#  [140 450 189]
#  [124 434 173]
#  [121 431 170]
#  [171 481 220]
#  [174 484 223]
#  [142 452 191]
#  [127 437 176]
#  [133 443 182]
#  [126 436 175]
#  [196 506 245]
#  [184 494 233]
#  [107 417 156]
#  [145 455 194]
#  [122 432 171]
#  [129 439 178]
#  [183 493 232]
#  [132 442 181]
#  [149 459 198]
#  [179 489 228]
#  [115 425 164]
#  [137 447 186]
#  [152 462 201]
#  [187 497 236]
#  [162 472 211]
#  [156 466 205]]
# 종이, 원달러환율:  
# [[ 175 1376]
#  [ 142 1343]
#  [ 146 1347]
#  [ 168 1369]
#  [ 103 1304]
#  [ 139 1340]
#  [ 123 1324]
#  [ 120 1321]
#  [ 170 1371]
#  [ 173 1374]
#  [ 141 1342]
#  [ 126 1327]
#  [ 132 1333]
#  [ 125 1326]
#  [ 195 1396]
#  [ 183 1384]
#  [ 106 1307]
#  [ 144 1345]
#  [ 121 1322]
#  [ 128 1329]
#  [ 182 1383]
#  [ 131 1332]
#  [ 148 1349]
#  [ 178 1379]
#  [ 114 1315]
#  [ 136 1337]
#  [ 151 1352]
#  [ 186 1387]
#  [ 161 1362]
#  [ 155 1356]]
# 프레딕트:  
# [[2076.299 ]
#  [2042.8688]
#  [2046.9181]
#  [2069.2053]
#  [2003.5579]
#  [2039.8326]
#  [2023.6776]
#  [2020.661 ]
#  [2071.2324]
#  [2074.2725]
#  [2041.8568]
#  [2026.6954]
#  [2032.7528]
#  [2025.6893]
#  [2096.565 ]
#  [2084.406 ]
#  [2006.5924]
#  [2044.8931]
#  [2021.6666]
#  [2028.7101]
#  [2083.3926]
#  [2031.7418]
#  [2048.9436]
#  [2079.3394]
#  [2014.6318]
#  [2036.7968]
#  [2051.9824]
#  [2087.4458]
#  [2062.1116]
#  [2056.034 ]]