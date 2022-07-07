# 과제
# activation : sigmoid, relu, linear 넣어라
# metrics 추가
# EarlyStopping 넣기
# 정확도 측정 (R2 사용)
# 성능 비교
# 감상문 2줄 이상! 뭐를 어케 했더니 뭐가 좋아졌고 안좋아졌다 써라

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
model.add(Dense(5, activation='relu', input_dim=13))
model.add(Dense(10, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1)) # relu는 마지막 out put에 넣으면 안됨

# 3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam', metrics=['mae'])

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

'''
# 그림을 그리자!
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

mpl.rcParams['font.family'] = 'malgun gothic'
mpl.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()
plt.title('보스턴//로스와 발리데이션 로스')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend()

print('걸린 시간 : ', end_time - start_time)

plt.show()
'''

# linear only
# loss:  18.226640701293945
# r2: 0.781933488319914

# with 1 relu (middle)
# loss:  12.171270370483398      
# r2: 0.8543809297257895

# with 2 relu
# loss:  14.652137756347656      
# r2: 0.8246994289462121

# with 3 relu
# loss:  13.014631271362305      
# r2: 0.844290794963973

# with 4 relu
# loss:  14.797404289245605      
# r2: 0.8229614150849752

# with 5 relu (include first layer)
# loss:  14.116393089294434      
# r2: 0.8311091415222092

# with 6 relu
# loss:  10.714729309082031      
# r2: 0.8718072307433771

# relu를 아예 안한 것 보다 한번 이상 넣은 것이 더 좋으며 많이 집어 넣는다고 무조건 성능 향상이 이루어지지는
# 않는다. 이것도 적절히 횟수를 튜닝하는 것이 필요해보인다.