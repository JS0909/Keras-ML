# 난이도 있음
# sigmoid 이진분류, Embedding 레이어 사용
# 시간 오래걸리니까 Conv1D 사용해도 됨

# ======================================================================
# There are 5 questions in this exam with increasing difficulty from 1-5.
# Please note that the weight of the grade for the question is relative
# to its difficulty. So your Category 1 question will score significantly
# less than your Category 5 question.
#
# Don't use lambda layers in your model.
# You do not need them to solve the question.
# Lambda layers are not supported by the grading infrastructure.
#
# You must use the Submit and Test button to submit your model
# at least once in this category before you finally submit your exam,
# otherwise you will score zero for this category.
# ======================================================================
#
# NLP QUESTION
#
# Build and train a classifier for the sarcasm dataset.
# The classifier should have a final layer with 1 neuron activated by sigmoid as shown.
# It will be tested against a number of sentences that the network hasn't previously seen
# and you will be scored on whether sarcasm was correctly detected in those sentences.

import time
import json
import tensorflow as tf
import numpy as np
import urllib
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.layers import Dense, Conv1D, GRU, Embedding, Flatten
from tensorflow.python.keras.models import Sequential

def solution_model():
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/sarcasm.json'
    urllib.request.urlretrieve(url, 'sarcasm.json')

    # DO NOT CHANGE THIS CODE OR THE TESTS MAY NOT WORK
    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type='post'
    padding_type='post'
    oov_tok = "<OOV>"
    training_size = 20000

    sentences = []
    labels = []
    # YOUR CODE HERE
    # json 파일 레이블만 빼서 labels에 넣어주면 됨
    with open('sarcasm.json', 'r') as f:
        files = json.load(f)
        for this in files:
            sentences.append(this['headline'])
            labels.append(this['is_sarcastic'])
    
    # print(len(sentences))   # 26709
    x_train = sentences[:training_size]
    y_train = labels[:training_size]
    
    x_test = sentences[training_size:]
    y_test = labels[training_size:]
    
    print(len(x_train), len(y_train))   # 20000 20000
    print(len(x_test), len(y_test))     # 6709 6709
    
    token = Tokenizer(oov_token=oov_tok)
    token.fit_on_texts(x_train)
    x_train = token.texts_to_sequences(x_train)
    x_test = token.texts_to_sequences(x_test)
    # print(token.word_index)

    x_train = pad_sequences(x_train, padding=padding_type, maxlen=max_length, truncating=trunc_type)
    x_test = pad_sequences(x_test, padding=padding_type, maxlen=max_length, truncating=trunc_type)
    
    # model = Sequential()
    # model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length))
    # model.add(Conv1D(10, 2, activation='relu'))
    # model.add(Flatten())
    # model.add(Dense(32, activation='relu'))
    # model.add(Dense(16, activation='relu'))
    # model.add(Dense(1, activation='sigmoid'))
    
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
        tf.keras.layers.Conv1D(20, 2, activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    start = time.time()
    model.fit(x_train, np.array(y_train), epochs=10, batch_size=20)
    end = time.time()
    
    loss, acc = model.evaluate(x_test, np.array(y_test))
    print('loss:', loss, 'acc:', acc)
    print(f'took:{end-start:.3} sec.')
    # loss: 0.6306837797164917 acc: 0.8010135889053345
    # took:50.4 sec.

    return model


# Note that you'll need to save your model as a .h5 like this.
# When you press the Submit and Test button, your saved .h5 model will
# be sent to the testing infrastructure for scoring
# and the score will be returned to you.
if __name__ == '__main__':
    model = solution_model()
    model.save("mymodel.h5")
