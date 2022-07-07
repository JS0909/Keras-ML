from sklearn.datasets import fetch_california_housing
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


# 1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
x_train, x_test, y_train, y_test =  train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)


# 2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=8))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(50))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))


# 3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam')
hist = model.fit(x_train, y_train, epochs=700, batch_size=100
          , validation_split=0.25)


# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x)

r2 = r2_score(y, y_predict)
print('r2스코어 : ', r2)


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
plt.title('로스와 발리데이션 로스')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend()
plt.show()