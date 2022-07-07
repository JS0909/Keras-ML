import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

# 1. 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10],
              [1,1,1,1,2,1.3,1.4,1.5,1.6,1.4],
              [9,8,7,6,5,4,3,2,1,0]])
y = np.array([11,12,13,14,15,16,17,18,19,20])

print(x.shape, y.shape) # (3, 10) (10,)
x = x.T # 행렬 전환
print(x.shape) # (10, 3)