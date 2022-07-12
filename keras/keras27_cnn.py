from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten

model = Sequential()
# model.add(Dense(units=10, input_shape=(1,))) # (batch_size, input_dim)
# model.summary() # (input_dim + bias) * units = summary Param (Dense모델)


model.add(Conv2D(filters=10, kernel_size=(3,3), # output (N, 5, 8, 10)
                 strides=1,
                 input_shape=(7, 10, 1))) # (batch_size전체데이터갯수, rows, columns, channels) = (행, 가로, 세로, 겹쳐진 장수)
# (몇장, 이미지가로, 세로, 흑백1 컬러3), 몇장은 빼고 씀 행무시, kernel_size=(n,n) n바이n짜리 이미지로 자르겠다
# strides=간격, 옆으로 몇칸 이동해서 연산할건지
# 필터의 갯수만큼 아웃풋 늘어남
# CNN : input (N, rows, columns, channels), output (N, rows, columns, filters)
# DNN : input (N, input_dim), output (N, unit)
model.add(Conv2D(7, (2,2), activation='relu')) # output (N, 5, 5, 7)
model.add(Flatten()) # (N, 175)
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary() # (kernel_size * channels + bias) * filters = summary Param (CNN 모델)

# 특성값을 키우고 shape는 작아짐