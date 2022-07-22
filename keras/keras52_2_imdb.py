from keras.datasets import imdb
from statistics import mean
import numpy as np

(x_train, y_train), (x_test, y_test) = imdb.load_data(
    num_words=10000 # 테스트 스플릿 안잡아주면 디폴트로 잡아 넘겨줌
)

print(x_train)
print(x_train.shape, x_test.shape) # (25000,) (25000,)
print(y_train) # [1 0 0 ... 0 1 0] // 이진분류
print(np.unique(y_train, return_counts=True))
print(len(np.unique(y_train))) # 2

# print(x_train[0].shape) // array형태가 아니라서 안먹음 len으로 확인해야됨
print(len(x_train[0]))
print(len(x_train[1]))

print('뉴스기사의 최대길이: ', max(len(i) for i in x_train)) # 2494
print('뉴스기사의 평균길이: ', mean(len(i) for i in x_train)) # 238.71364
print('뉴스기사의 평균길이: ', sum(map(len, x_train)) / len(x_train)) # 238.71364

# 전처리
from keras.preprocessing.sequence import pad_sequences

x_train = pad_sequences(x_train, padding='pre', maxlen=2000, truncating='pre')
x_test = pad_sequences(x_test, padding='pre', maxlen=2000, truncating='pre')

print(x_train.shape, y_train.shape) # (25000, 2000) (25000, 2)
print(x_test.shape, y_test.shape)  # (25000, 2000) (25000, 2)

# 2. 모델
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv1D, LSTM, Embedding

model = Sequential()
model.add(Embedding(input_dim=2, output_dim=10, input_length=2000))
model.add(LSTM(32))
model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# model.summary()

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=1, batch_size=1000)

# 4. 평가, 예측
acc = model.evaluate(x_test, y_test)[1]
pred = model.predict(x_test)
print('acc: ', acc)
print('결과: ', pred)

# acc:  0.5
# 결과:  [[0.4993433]
#  [0.4993433]
#  [0.4993433]
#  ...
#  [0.4993433]
#  [0.4993433]
#  [0.4993433]]