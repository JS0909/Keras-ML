from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np

# 1. 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])

# 2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(3, activation='relu'))
model.add(Dense(4, activation='sigmoid'))
model.add(Dense(2, activation='relu'))
model.add(Dense(1))

model.summary() # 총 연산량 확인

# 인풋레이어 안에 bias 노드를 1개씩 추가해서 계산한다
# 위에 모델을 예로 들면 처음에 (5+1)*3번 연산을 하는 셈