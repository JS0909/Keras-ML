from sklearn.datasets import load_boston
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time


# 1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target
x_train, x_test, y_train, y_test =  train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)


# 2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=13))
model.add(Dense(10))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))


# 3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam')

start_time = time.time() # 현재 시간 리턴
hist = model.fit(x_train, y_train, epochs=91, batch_size=1,
                 validation_split=0.2,
                 verbose=1)
# validation은 검증용이라서 실제 훈련에 영향 안미치므로 적게 잡아도 된다
end_time = time.time()

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss: ', loss)
y_predict = model.predict(x)
r2 = r2_score(y, y_predict)
print('r2:', r2)
'''
print('==============================================')
print(hist) # <tensorflow.python.keras.callbacks.History object at 0x0000026193E37640> model.fit은 주소를 리턴함
print('==============================================')
print(hist.history) # 그래서 .history로 딕셔너리 자체를 리턴 시켜서 봄
                    # {'loss': [196.9160919189453, 109.63496398925781, 105.44699096679688, 92.84196472167969, 91.95366668701172, 88.09672546386719, 77.24915313720703, 80.857666015625, 77.52342987060547, 73.37767028808594, 71.57038879394531], 
                    #'val_loss': [139.7096405029297, 82.95533752441406, 82.9881820678711, 82.4740219116211, 79.36639404296875, 82.91313934326172, 95.74589538574219, 
                    # 82.1336441040039, 96.72431182861328, 75.15653991699219, 71.48831939697266]}
                    # 딕셔너리 형태, key와 데이터로 구성, 위에꺼는 key value가 loss와 val_loss가 있는 것
print('==============================================')
print(hist.history['loss'])
print('\n ', hist.history['val_loss'])
'''

# 그림을 그리자!
import matplotlib as mpl # 폰트 지정하는 rcParams를 위해서 임포트
# import matplotlib.pyplot as plt // 상위 matplotlib에 포함되어 있어서 임포트 따로 안해도 되긴 함

mpl.rcParams['font.family'] = 'malgun gothic' # 폰트 지정
mpl.rcParams['axes.unicode_minus'] = False

mpl.figure(figsize=(9,6))
mpl.plot(hist.history['loss'], marker='.', c='red', label='loss')
# x가 1일 때 hist.history[1], 2일때 hist.history[2] ... 인 셈
mpl.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
mpl.grid()
mpl.title('보스턴//로스와 발리데이션 로스')
mpl.ylabel('loss')
mpl.xlabel('epochs')
# mpl.legend(loc='upper right')
mpl.legend()

print('걸린 시간 : ', end_time - start_time) # 시간 체킹

mpl.show()