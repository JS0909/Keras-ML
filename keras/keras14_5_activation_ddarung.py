# Dacon 따릉이 문제풀이
import numpy as np
import pandas as pd 
from pandas import DataFrame 
from sklearn.datasets import fetch_california_housing
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.python.keras.callbacks import EarlyStopping


# 1. 데이터
path = './_data/ddarung/'
train_set = pd.read_csv(path+'train.csv',index_col=0) # index_col = n : n번째 칼럼을 인덱스로 인식
# print(train_set)
# print(train_set.shape) # (1459, 10)

test_set = pd.read_csv(path+'test.csv', index_col=0)
# print(test_set)
# print(test_set.shape) # (715, 9)

### 결측치 처리(일단 제거로 처리) ###
print(train_set.info())
print(train_set.isnull().sum()) # 결측치 전부 더함
train_set = train_set.dropna() # nan 값(결측치) 자동으로 0으로 만듦
print(train_set.isnull().sum()) # 없어졌는지 재확인

x = train_set.drop(['count'], axis=1) # axis = 0은 열방향으로 쭉 한줄(가로로 쭉), 1은 행방향으로 쭉 한줄(세로로 쭉)
y = train_set['count']

print(x.shape, y.shape) # (1328, 9) (1328,)

# trainset과 testset의 분리
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)

# 2. 모델 구성
model = Sequential()
model.add(Dense(10, activation='relu', input_dim=9))
model.add(Dense(30
                , activation='relu'
                ))
model.add(Dense(50
                , activation='relu'
                ))
model.add(Dense(20
                , activation='relu'
                ))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
Es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50, restore_best_weights=True)
log = model.fit(x_train, y_train, epochs=1000, batch_size=50, callbacks=[Es], validation_split=0.25)

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss: ', loss)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2: ', r2)

#======그래프======
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

mpl.rcParams['font.family'] = 'malgun gothic'
mpl.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(9,6))
plt.plot(log.history['loss'], marker='.', c='red', label='loss')
plt.plot(log.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()
plt.title('따릉이//로스와 발리데이션 로스')
plt.ylabel('loss')
plt.xlabel('epochs')
# plt.legend()
plt.show()
#======그래프======


# 5. 제출 준비
submission = pd.read_csv(path + 'submission.csv', index_col=0)
y_submit = model.predict(test_set)
submission['count'] = y_submit
submission.to_csv(path + 'submission.csv', index=True) # 실제 파일에 push해주는 과정
                                                        # False: 존재하는 index자리 삭제하고 넣겠다
                                                        # True: 존재하는 index자리 유지하고 넣겠다
                                                               
# linear only
# loss:  [mse: 2809.820068359375, mae: 39.7796745300293]
# r2:  0.5576589362116394

# with 2 relu
# loss:  [mse: 2680.8916015625, mae: 38.65082931518555]
# r2:  0.6034229453546867

# with 3 relu
# loss:  [mse: 2608.839111328125, mae: 39.5834846496582]
# r2:  0.6103558331255308

# with 4 relu
# loss:  [mse: 2624.9033203125, mae: 36.541927337646484]
# r2:  0.598056884925878

# relu 세개정도 사용했을 때 가장 효율적으로 보임