import numpy as np
from sklearn import metrics
from sklearn.datasets import load_breast_cancer
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential
from sklearn.model_selection import train_test_split
import tensorflow as tf

import time

gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
if(gpus):
    aaa = 'GPU'
else:
    aaa = 'CPU'
    # GPU 실행 여부에따라 마지막 출력 명에 GPU인지, CPU 출력 추가

# 1. 데이터
datasets = load_breast_cancer()
# print(datasets)
# print(datasets.DESCR) // Instances: 569, Attributes: 30
# print(datasets.feature_names)

x = datasets.data # datasets['data']
y = datasets.target # datasets['target'] // key value니까 이렇게도 가능
print(x.shape, y.shape) # (569, 30) (569,)

# print(x)
# print(y)

x_train, x_test, y_train, y_test =  train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)


# 2. 모델구성
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
# binary_crossentropy : (이진분류) 1 or 0 으로 반올림시키는 방식으로 sigmoid로 나온 아웃풋 loss 값 계산해서 verbose에서 보도록 컴파일 하겠다
# metrics : 훈련하면서 정확도를 verbose에서 ['평가방식']으로 보도록 컴파일하겠다

from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=200, mode='min', verbose=1, restore_best_weights=True)
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=1,
                callbacks=[earlyStopping],
                validation_split=0.25)
end_time = time.time()

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)
###### [과제 1.] accuracy score 완성시키기
from sklearn.metrics import r2_score, accuracy_score
# r2 = r2_score(y_test, y_predict) # 이진분류라서 r2 안씀
y_predict = np.round(y_predict,0)
acc = accuracy_score(y_test, y_predict)
print('acc스코어 : ', acc)

print(aaa, '걸린 시간: ', end_time-start_time)

# GPU 사용시 걸린 시간:  170.12930369377136
# CPU 사용시 걸린 시간:  28.693891286849976

# 작은 데이터나 수치형 데이터는 CPU가 더 빠를 수 있다