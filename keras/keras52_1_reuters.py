from statistics import mean
from keras.datasets import reuters
import numpy as np
import pandas as pd

(x_train, y_train), (x_test, y_test) = reuters.load_data(
    num_words=10000, test_split=0.2
    # 빈도수 많은 순으로 10000개 단어를 가져온다, 나머지는 0으로 채우는듯
)

print(x_train)
print(x_train.shape, x_test.shape) # (8982,) (2246,)
print(y_train) # [ 3  4  3 ... 25  3 25]
print(np.unique(y_train, return_counts=True))
print(len(np.unique(y_train))) # 46개 단어가 있음

print(type(x_train), type(y_train)) # <class 'numpy.ndarray'> <class 'numpy.ndarray'>
print(type(x_train[0]))             # <class 'list'> 일정하지 않은 리스트, 패딩 필요
# print(x_train[0].shape) // array형태가 아니라서 안먹음 len으로 확인해야됨
print(len(x_train[0])) # 87
print(len(x_train[1])) # 56
# 문장별로 길이(단어갯수)가 제각각임을 확인했다, 패딩 필요

print('뉴스기사의 최대길이: ', max(len(i) for i in x_train)) # 2376
print('뉴스기사의 평균길이: ', mean(len(i) for i in x_train)) # 145.5398574927633
print('뉴스기사의 평균길이: ', sum(map(len, x_train)) / len(x_train)) # 145.5398574927633

# 전처리
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

x_train = pad_sequences(x_train, padding='pre', maxlen=2000, truncating='pre') # (8982,) -> (8982, 100)
x_test = pad_sequences(x_test, padding='pre', maxlen=2000, truncating='pre') # (2246,) -> (2246, 100)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(x_train.shape, y_train.shape) # (8982, 100) (8982, 46)
print(x_test.shape, y_test.shape) # (2246, 100) (2246, 46)

# 2. 모델
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv1D, LSTM, Embedding

model = Sequential()
model.add(Embedding(input_dim=47, output_dim=10, input_length=2000))
model.add(LSTM(10))
model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(46, activation='softmax'))
# model.summary()

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=10, batch_size=5000)

# 4. 평가, 예측
acc = model.evaluate(x_test, y_test)[1]
pred = model.predict(x_test)
pred = np.argmax(pred, axis=1)
print('acc: ', acc)
print('결과: ', pred)

# acc:  0.21104185283184052
# 결과:  [3 3 3 ... 3 3 3]