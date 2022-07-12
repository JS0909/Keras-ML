import numpy as np
from sklearn import metrics
from sklearn.datasets import load_breast_cancer
from tensorflow.python.keras.layers import Dense, Input
from tensorflow.python.keras.models import Sequential, Model, load_model
from sklearn.model_selection import train_test_split
import datetime
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score

# 1. 데이터
datasets = load_breast_cancer()

x = datasets.data # datasets['data']
y = datasets.target # datasets['target'] // key value니까 이렇게도 가능
print(x.shape, y.shape) # (569, 30) (569,)

x_train, x_test, y_train, y_test =  train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)


# 2. 모델구성
# 시퀀셜
model = Sequential()
model.add(Dense(5, activation='relu', input_dim=30))
model.add(Dense(50, activation=None)) # activation defualt = None (linear)
model.add(Dense(40, activation = 'relu')) # !중간에서만 쓸 수 있다, 평타 85% 이상 개좋은 놈
model.add(Dense(100, activation='sigmoid'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid')) # 0과 1 사이의 유리수로 최종 out put이 저장된다

# 3. 컴파일, 훈련
model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['accuracy', 'mse']) 
date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M')
filepath = './_ModelCheckPoint/k25/04/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
earlyStopping = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1, restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1,
                      save_best_only=True, filepath= "".join([filepath, 'k25_',date, '_', filename]))
hist = model.fit(x_train, y_train, epochs=1000, batch_size=50,
                callbacks=[earlyStopping, mcp],
                validation_split=0.25)


# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)
y_predict = np.round(y_predict,0)
acc = accuracy_score(y_test, y_predict)
print('acc스코어 : ', acc)
