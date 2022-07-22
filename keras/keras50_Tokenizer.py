from keras.preprocessing.text import Tokenizer

text = '나는 진짜 매우 매우 맛있는 밥을 엄청 마구 마구 마구 먹었다.'

token = Tokenizer()
token.fit_on_texts([text]) # 리스트형태로 받음, 순서대로 인덱스가 잡히면서 반복횟수가 많은 순서로 앞으로 정리한다

print(token.word_index) # 인덱스별로 출력해본다
# {'마구': 1, '매우': 2, '나는': 3, '진짜': 4, '맛있는': 5, '밥을': 6, '엄청': 7, '먹었다': 8}

x = token.texts_to_sequences([text])
print(x)

# multicollinearity 방지 ex) 1+1=2, 마구+마구=매우
from tensorflow.python.keras.utils.np_utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# x = to_categorical(x)
# print(x)

ohe = OneHotEncoder() # 원핫 고쳐
x = np.array(x)
x = x.reshape(-1, 1)
x = ohe.fit_transform(x).toarray()
print(x)
# [[0. 0. 1. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 1. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 1. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 1. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 1. 0.]
#  [1. 0. 0. 0. 0. 0. 0. 0.]
#  [1. 0. 0. 0. 0. 0. 0. 0.]
#  [1. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 1.]]