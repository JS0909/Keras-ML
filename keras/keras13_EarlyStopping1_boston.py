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

from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=30, mode='min', verbose=1, restore_best_weights=True)
# monitor에서 제공하는 것은 val_loss, loss 정도이고 R2는 제공 안함
# mode auto면 loss계열은 자동으로 최소, accuracy계열은 자동으로 최대 찾음

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size=1,
                 validation_split=0.2,
                 callbacks=[earlyStopping],
                 verbose=1)
end_time = time.time()

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss: ', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2:', r2)

print('==============================================')
print(hist)
print('==============================================')
print(hist.history)
print('==============================================')
print('loss: ',hist.history['loss'])
print('\n val_loss: ', hist.history['val_loss'])

# 그림을 그리자!
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

mpl.rcParams['font.family'] = 'malgun gothic'
mpl.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
# x가 1일 때 hist.history[1], 2일때 hist.history[2] ... 인 셈
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()
plt.title('보스턴//로스와 발리데이션 로스')
plt.ylabel('loss')
plt.xlabel('epochs')
# plt.legend(loc='upper right')
plt.legend()

print('걸린 시간 : ', end_time - start_time) # 시간 체킹

plt.show()


#============================================
# loss:  20.629039764404297
# r2: 0.7531907652550968