from keras.preprocessing.text import Tokenizer
import numpy as np

# 1. 데이터
docs = ['너무 재밋어요', '참 최고에요', '참 잘 만든 영화에요', '추천하고 싶은 영화입니다.', '한 번 더 보고 싶네요', '글세요', '별로에요',
        '생각보다 지루해요', '연기가 어색해요', '재미없어요', '너무 재미없다', '참 재밋네요', '민수가 못 생기긴 했어요', '안결 혼해요']

# 긍정 1, 부정 0
labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,1,0]) # (14,)
# y값임

token = Tokenizer()
token.fit_on_texts(docs) # docs를 기준으로 tokenizer를 fit한다
print(token.word_index)
# {'참': 1, '너무': 2, '재밋어요': 3, '최고에요': 4, '잘': 5, '만든': 6, 
# '영화에요': 7, '추천하고': 8, '싶은': 9, '영화입니다': 10, '한': 11, '번': 12, '더': 13, '보고': 14, '싶네요': 15, '글세요': 16, 
# '별로에요': 17, '생각보다': 18, '지루해요': 19, '연기가': 20, '어색해요': 21, '재미없어요': 22, '재미없다': 23, '재밋네요': 24, '민수가': 25, 
# '못': 26, '생기긴': 27, '했어요': 28, '안결': 29, '혼해요': 30}
# 0번 인덱스를 사용하지 않는 이유는 가장 긴 값에 맞춰 짧은 것들에 0으로 채워 길이를 맞추기 위함이다

x = token.texts_to_sequences(docs) # 단어들을 인덱스 번호로 교체함
print(x)
# [[2, 3], [1, 4], [1, 5, 6, 7], [8, 9, 10], [11, 12, 13, 14, 15], [16], [17], [18, 19], [20, 21], [22], [2, 23], [1, 24], [25, 26, 27, 28], [29, 30]]

from keras.preprocessing.sequence import pad_sequences
pad_x = pad_sequences(x, padding='pre', maxlen=5) 
# padding : x의 데이터에 0을 채우거나 데이터를 잘라서 뒤에 오는 숫자만큼 크기를 맞춤
# 시계열 데이터이기 때문에 통상 앞부분에 0을 채우는 것이 좋다
# pre : 앞에 0채움 / post : 뒤에 0채움

print(pad_x)
# [[ 0  0  0  2  3]
#  [ 0  0  0  1  4]
#  [ 0  1  5  6  7]
#  [ 0  0  8  9 10]
#  [11 12 13 14 15]
#  [ 0  0  0  0 16]
#  [ 0  0  0  0 17]
#  [ 0  0  0 18 19]
#  [ 0  0  0 20 21]
#  [ 0  0  0  0 22]
#  [ 0  0  0  2 23]
#  [ 0  0  0  1 24]
#  [ 0 25 26 27 28]
#  [ 0  0  0 29 30]]

print(pad_x.shape) # (14, 5)

word_size = len(token.word_index)
print('word_size: ', word_size)       # 단어사전의 갯수: 30 + 1(패딩 0짜리 포함)

print(np.unique(pad_x, return_counts=True))
# (array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]), // + 0(패딩값)
# array([37,  3,  2,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1], dtype=int64))


# 2. 모델
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv1D, LSTM, Embedding

model = Sequential() # 현재 인풋은 원핫을 하지 않은 (14, 5)
                     # (단어사전의 총 갯수, 원하는 아웃풋노드 갯수, 단어들의 길이)
model.add(Embedding(input_dim=31, output_dim=10, input_length=5)) # input_length의 갯수는 모르면 명시 안해도 잡아준다
# (N, 5, 10) timestep 5개짜리가 10개 있음
# param 갯수 = 단어갯수*아웃풋노드 갯수 // 단어 갯수 31개 = 패딩 단어 0이 추가되서
# model.add(Embedding(31, 10)) # 이렇게 써줘도 된다
# model.add(Embedding(31, 10, input_lenght=5)) # 이렇게 써줘도 된다
# model.add(Embedding(31, 10, 5)) # 이건 안됨
# input_dim이 틀려도 돌아는 간다. 대신 적게 입력하면 그만큼 적은 단어만 사용해서 훈련하는 셈. 데이터 버리는 게 됨.

model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid')) # label(y값) 정의를 0부정 또는 1긍정으로 했음
model.summary()

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(pad_x, labels, epochs=20, batch_size=16)

# 4. 평가, 예측
acc = model.evaluate(pad_x, labels)[1] # compile에서 metrics 사용했으니까 [0]은 loss, [1]은 acc
print('acc: ', acc)

# acc:  0.9285714030265808