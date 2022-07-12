from sklearn.datasets import load_boston
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

# 1. 데이터
datasets = load_boston()
x, y = datasets.data, datasets.target
x_train, x_test, y_train, y_test =  train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델구성
model = Sequential()
model.add(Dense(64, input_dim=13))
model.add(Dense(32))
model.add(Dense(16, activation='relu'))
model.add(Dense(8))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam')

import datetime
date = datetime.datetime.now()
print(date) # 2022-07-07 17:24:42.792511 datetime형
date = date.strftime('%m%d_%H%M')
print(date) # 0707_1723 str형

filepath = './_ModelCheckPoint/k24/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
# epoch 4자리까지, val_loss 소수점 4자리까지

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1,
                      save_best_only=True, filepath= "".join([filepath, 'k24_',date, '_', filename])) # 빈문자열 만든 후 join 안의 것들을 쭉 넣겠다
log = model.fit(x_train, y_train, epochs=100, batch_size=1, callbacks=[es, mcp], validation_split=0.2, verbose=1)

# 4. 평가, 예측
print("============== 1. 기본 출력 ==================")
loss = model.evaluate(x_test, y_test)
print('loss: ', loss)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2:', r2)

'''
print("============== 2. load_model 출력 ==================")
model2 = load_model('./_save/keras24_3_save_model.h5')
loss2 = model2.evaluate(x_test, y_test)
print('loss2: ', loss2)
y_predict2 = model.predict(x_test)
r2 = r2_score(y_test, y_predict2)
print('r2:', r2)

print("============== 3. ModelCheckPoint 출력 ==================")
model3 = load_model('./_ModelCheckPoint/keras24_ModelCheckPoint3.hdf5')
loss3 = model3.evaluate(x_test, y_test)
print('loss3: ', loss3)
y_predict2 = model.predict(x_test)
r2 = r2_score(y_test, y_predict2)
print('r2:', r2)

'''
