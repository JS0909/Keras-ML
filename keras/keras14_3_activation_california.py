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
x_train, x_test, y_train, y_test =  train_test_split(x, y, train_size=0.7, shuffle=True, random_state=66)

# 2. 모델구성
model = Sequential()
model.add(Dense(20, activation='relu', input_dim=8))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(70, activation='relu'))
model.add(Dense(50, activation='relu'))
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
# loss :  [mse: 0.8611065149307251, mae: 0.6064989566802979]
# r2스코어 :  0.3724495049420907

# with 1 relu (middle)
# loss :  [mse: 0.488613486289978, mae: 0.5227382183074951]
# r2스코어 :  0.6439117981275244
  
# with 2 relu
# loss :  [mse: 0.6959879398345947, mae: 0.6162382364273071]
# r2스코어 :  0.49278292122645606

# with 3 relu
# loss :  [mse: 0.7207165360450745, mae: 0.6578399538993835]
# r2스코어 :  0.47476151022840285

# with 4 relu
# loss :  [mse: 0.4799359142780304, mae: 0.5071946978569031]
# r2스코어 :  0.6502359389500392

# with 5 relu (include first layer)
# loss :  [mse: 0.4737195074558258, mae: 0.5066453814506531]
# r2스코어 :  0.6547662445023353

# with 6 relu
# loss :  [mse: 0.479023814201355, mae: 0.5100365877151489]
# r2스코어 :  0.6509004738470596