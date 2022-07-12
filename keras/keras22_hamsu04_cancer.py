import numpy as np
from sklearn import metrics
from sklearn.datasets import load_breast_cancer
from tensorflow.python.keras.layers import Dense, Input
from tensorflow.python.keras.models import Sequential, Model
from sklearn.model_selection import train_test_split
import time
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 1. 데이터
datasets = load_breast_cancer()
# print(datasets)
# print(datasets.DESCR) // Instances: 569, Attributes: 30
# print(datasets.feature_names)

x = datasets.data # datasets['data']
y = datasets.target # datasets['target'] // key value니까 이렇게도 가능
print(x.shape, y.shape) # (569, 30) (569,)

x_train, x_test, y_train, y_test =  train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler() 
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(np.min(x_train)) # 0.0
print(np.max(x_train)) # 1.0
print(np.min(x_test))
print(np.max(x_test))

# 2. 모델구성
# 시퀀셜
# model = Sequential()
# model.add(Dense(5, activation='relu', input_dim=30))
# model.add(Dense(50, activation=None)) # activation defualt = None (linear)
# model.add(Dense(40, activation = 'relu')) # !중간에서만 쓸 수 있다, 평타 85% 이상 개좋은 놈
# model.add(Dense(100, activation='sigmoid'))
# model.add(Dense(20, activation='relu'))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(1, activation='sigmoid')) # 0과 1 사이의 유리수로 최종 out put이 저장된다

# 함수모델
input1 = Input(shape=(30,))
dense1 = Dense(5, activation='relu')(input1)
dense2 = Dense(50)(dense1)
dense3 = Dense(40, activation='relu')(dense2)
dense4 = Dense(100, activation='sigmoid')(dense3)
dense5 = Dense(20, activation='relu')(dense4)
dense6 = Dense(10, activation='relu')(dense5)
output1 = Dense(1, activation='sigmoid')(dense6)
model = Model(inputs=input1,outputs=output1)

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

# 시퀀셜 StandardScaler
# loss :  [0.07493800669908524, 0.9736841917037964, 0.020953446626663208]
# acc스코어 :  0.9736842105263158
# 걸린 시간:  2.553802967071533

# 함수모델
# loss :  [0.15105924010276794, 0.9561403393745422, 0.03777826204895973]
# acc스코어 :  0.956140350877193
# 걸린 시간:  2.369398832321167