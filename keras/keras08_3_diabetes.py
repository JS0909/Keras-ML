from sklearn.datasets import load_diabetes
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


# 1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
x_train, x_test, y_train, y_test =  train_test_split(x, y, train_size=0.9, shuffle=True, random_state=86)

print(x.shape) 
print(y.shape) 

# [실습] 아래를 완성할 것
# R2 0.62 이상


# 2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=10))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(10))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam')
model.fit(x_train, y_train, epochs=500, batch_size=5)

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x)

r2 = r2_score(y, y_predict)
print('r2스코어 : ', r2)

# loss :  1672.607177734375  
# r2스코어 :  0.5149952410421789