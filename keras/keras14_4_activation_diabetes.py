from sklearn.datasets import fetch_california_housing
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time

# 1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
x_train, x_test, y_train, y_test =  train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)


# 2. 모델구성
model = Sequential()
model.add(Dense(5, activation='relu', input_dim=8))
model.add(Dense(50, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))


# 3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam', metrics=['mae'])

from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=30, mode='min', verbose=1, restore_best_weights=True)
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=2000, batch_size=50,
                callbacks=[earlyStopping],
                validation_split=0.25)
end_time = time.time()

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)

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
plt.title('캘리포니아//로스와 발리데이션 로스')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend()
plt.show()
'''

# linear only
# loss :  [mse: 0.7418478727340698, mae: 0.5937469005584717]
# r2스코어 :  0.4678528727991814

# with 1 relu (middle)
# loss :  [mse: 0.7158827781677246, mae: 0.617363452911377]
# r2스코어 :  0.48647816826915813
  
# with 2 relu
# loss :  [mse: 0.4886036515235901, mae: 0.5166036486625671]
# r2스코어 :  0.6495115089308896

# with 3 relu
# loss :  [mse: 0.4493708908557892, mae: 0.4895772337913513]
# r2스코어 :  0.6776542966810682

# with 4 relu
# # loss :  [mse: 0.47432729601860046, mae: 0.5114820003509521]
# r2스코어 :  0.6597524358434084

# with 5 relu (include first layer)
# loss :  [mse: 0.4918108880519867, mae: 0.5181877613067627]
# r2스코어 :  0.647210961019032

# with 6 relu
# loss :  [mse: 0.510589063167572, mae: 0.5115727782249451]
# r2스코어 :  0.6337408648815703

# relu를 한번 사용했을때 성능이 눈에 띄게 좋아졌고 세번 사용했을때 제일 좋았음
# 2차함수 모양으로 loss와 R2 스코어가 바뀌고 있음
# 이 모델에서는 relu를 세번 사용해주는 것이 가장 적합하다고 판단됨