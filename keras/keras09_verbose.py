from sklearn.datasets import load_boston
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


# 1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target
x_train, x_test, y_train, y_test =  train_test_split(x, y, train_size=0.7, shuffle=True, random_state=66)

'''
print(x)
print(y)
print(x.shape) # (506, 13)
print(y.shape) # (506,)

print(datasets.feature_names) # 열 별로 이름 나옴
#['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO' 'B' 'LSTAT']
print(datasets.DESCR)
'''


# 2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=13))
model.add(Dense(10))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))


import time
# 3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam')

start_time = time.time() # 현재 시간 리턴
model.fit(x_train, y_train, epochs=50, batch_size=1,
          verbose=1) # 중간 과정 보일지 말지 정함 defualt는 1
# 인간 눈에 보여주기위해 딜레이를 걸기 때문에 verbose를 켤 경우 통상적으로 더 느려진다
# 2 : 프로그래스 바 제외
# 3이상 : epoch만 표시
end_time = time.time()


# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss: ', loss)

print('걸린 시간 : ', end_time - start_time) # 시간 체킹

'''
verbose 0 걸린 시간 :  8.758423805236816 / 출력 없다.
verbose 1 걸린시간 :  10.278096199035645 / 프로그래스바 o, loss o, epoch o
verbose 2 걸린 시간 :  8.228228092193604 / 프로그래스바 x, loss o, epoch o
verbose 3 걸린 시간 :  8.138976812362671 / 프로그래스바 x, loss x, epoch o
'''