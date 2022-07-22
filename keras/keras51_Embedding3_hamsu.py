from keras.preprocessing.text import Tokenizer
import numpy as np

# 1. 데이터
docs = ['너무 재밋어요', '참 최고에요', '참 잘 만든 영화에요', '추천하고 싶은 영화입니다.', '한 번 더 보고 싶네요', '글세요', '별로에요',
        '생각보다 지루해요', '연기가 어색해요', '재미없어요', '너무 재미없다', '참 재밋네요', '민수가 못 생기긴 했어요', '안결 혼해요']
x_predict = ['나는 형권이가 정말 재미없다 너무 정말']

# 긍정 1, 부정 0
labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,1,0]) # (14,)

token = Tokenizer(oov_token="<OOV>")
token.fit_on_texts(docs)
token.fit_on_texts(x_predict)

x = token.texts_to_sequences(docs)
y = token.texts_to_sequences(x_predict)

from keras.preprocessing.sequence import pad_sequences
pad_x = pad_sequences(x, padding='pre', maxlen=5, truncating='pre') # truncating 
pad_y = pad_sequences(y, padding='pre', maxlen=5, truncating='pre')

print(pad_x)
print(pad_x.shape) # (14, 5)

word_size = len(token.word_index)
print('word_size: ', word_size)       # 단어사전의 갯수: 30

print(np.unique(pad_x, return_counts=True))
# (array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]), 
# array([37,  3,  2,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1], dtype=int64))

# 2. 모델
from tensorflow.python.keras.models import Sequential, Input, Model
from tensorflow.python.keras.layers import Dense, Conv1D, LSTM, Embedding

# model = Sequential()
# model.add(Embedding(input_dim=31, output_dim=10, input_length=5))
# model.add(LSTM(32))
# model.add(Dense(1, activation='sigmoid'))

input1 = Input(shape=(5,))
emb = Embedding(input_dim = 31,output_dim=10, input_length=5)(input1)
lstm1 = LSTM(20)(emb)
dense1 = Dense(80, activation='relu')(lstm1)
dense2 = Dense(80)(dense1)
dense3 = Dense(90)(dense2)
dense4 = Dense(70, activation='relu')(dense3)
dense5 = Dense(50, activation='relu')(dense4)
output1 = Dense(1, activation='sigmoid')(dense5)
model = Model(inputs=input1,outputs=output1)
model.summary()

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(pad_x, labels, epochs=20, batch_size=16)

# 4. 평가, 예측
acc = model.evaluate(pad_x, labels)[1]
pred = model.predict(x_predict)
print('acc: ', acc)
print('결과: ', pred)


#### [실습] ####
# x_predict = '나는 형권이가 정말 재미없다 너무 정말'

# 결과는? 긍정? 부정?