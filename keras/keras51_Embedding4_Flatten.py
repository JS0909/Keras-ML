from keras.preprocessing.text import Tokenizer
import numpy as np

# 1. 데이터
docs = ['너무 재밋어요', '참 최고에요', '참 잘 만든 영화에요', '추천하고 싶은 영화입니다.', '한 번 더 보고 싶네요', '글세요', '별로에요',
        '생각보다 지루해요', '연기가 어색해요', '재미없어요', '너무 재미없다', '참 재밋네요', '민수가 못 생기긴 했어요', '안결 혼해요']

# 긍정 1, 부정 0
labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,1,0]) # (14,)

token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)
# {'참': 1, '너무': 2, '재밋어요': 3, '최고에요': 4, '잘': 5, '만든': 6, 
# '영화에요': 7, '추천하고': 8, '싶은': 9, '영화입니다': 10, '한': 11, '번': 12, '더': 13, '보고': 14, '싶네요': 15, '글세요': 16, 
# '별로에요': 17, '생각보다': 18, '지루해요': 19, '연기가': 20, '어색해요': 21, '재미없어요': 22, '재미없다': 23, '재밋네요': 24, '민수가': 25, 
# '못': 26, '생기긴': 27, '했어요': 28, '안결': 29, '혼해요': 30}

x = token.texts_to_sequences(docs)
print(x)
# [[2, 3], [1, 4], [1, 5, 6, 7], [8, 9, 10], [11, 12, 13, 14, 15], [16], [17], [18, 19], [20, 21], [22], [2, 23], [1, 24], [25, 26, 27, 28], [29, 30]]
# 0번 인덱스를 사용하지 않는 이유는 가장 긴 값에 맞춰 짧은 것들에 0으로 채워 길이를 맞추기 위함이다

from keras.preprocessing.sequence import pad_sequences
pad_x = pad_sequences(x, padding='pre', maxlen=5) 
# x의 데이터에 0을 채우거나 데이터를 잘라서 뒤에 오는 숫자만큼 크기를 맞춤
# 시계열 데이터이기 때문에 통상 앞부분에 0을 채우는 것이 좋다

print(pad_x)

print(pad_x.shape) # (14, 5)

word_size = len(token.word_index)
print('word_size: ', word_size)       # 단어사전의 갯수: 30

print(np.unique(pad_x, return_counts=True))

# 2. 모델
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv1D, LSTM, Embedding, Flatten

model = Sequential()
model.add(Embedding(input_dim=31, output_dim=10, input_length=5))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(pad_x, labels, epochs=20, batch_size=16)

# 4. 평가, 예측
acc = model.evaluate(pad_x, labels)[1]
print('acc: ', acc)
