# 4차원 데이터 3차원으로 reshape하라고 함 shape 조심
# Dense, Conv1D 혹은 3차원짜리 input layer 이후 reshape layer 사용 후 Conv2D
# MNIST or Fashion MNIST


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
# Basic Datasets Question
#
# Create a classifier for the Fashion MNIST dataset
# Note that the test will expect it to classify 10 classes and that the
# input shape should be the native size of the Fashion MNIST dataset which is
# 28x28 monochrome. Do not resize the data. Your input layer should accept
# (28,28) as the input shape only. If you amend this, the tests will fail.
#
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv1D, Flatten
import numpy as np
import tensorflow as tf
from keras.utils import to_categorical
import time


def solution_model():
    fashion_mnist = tf.keras.datasets.fashion_mnist

    # YOUR CODE HERE
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    
    y_train= to_categorical(y_train)
    y_test= to_categorical(y_test)
    
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, strides=1, padding='valid', input_shape=(28, 28)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32))
    model.add(Dense(10, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    start = time.time()
    model.fit(x_train, y_train, epochs=20, batch_size=128, validation_split=0.2)
    end = time.time()
        
    loss, acc = model.evaluate(x_test, y_test)   
    print('loss:', loss, 'acc:', acc)
    print(f'took:{end-start:.3} sec.')
    # loss: 0.4516112804412842 acc: 0.8543999791145325
    # took:35.8 sec.
     
    return model


# Note that you'll need to save your model as a .h5 like this.
# When you press the Submit and Test button, your saved .h5 model will
# be sent to the testing infrastructure for scoring
# and the score will be returned to you.
if __name__ == '__main__':
    model = solution_model()
    model.save("mymodel.h5")
