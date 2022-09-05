import tensorflow as tf
import keras
import numpy as np

tf.compat.v1.set_random_seed(123)

# 1. data
from keras.datasets import mnist # tf1에서는 별도로 keras 설치해야함
(x_train, y_train), (x_test, y_test) = mnist.load_data()

from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255. # 스케일링 해준 셈, 1~255 사이즈이기 때문

# 2. model
x = tf.compat.v1.placeholder(tf.float32, [None, 28, 28, 1]) # input_shape
y = tf.compat.v1.placeholder(tf.float32, [None, 10])

w1 = tf.compat.v1.get_variable('w1', shape=[2, 2, 1, 64]) # cnn에서는 4차원으로 연산 [커널사이즈(AxA), 채널(컬러), 아웃풋노드(필터)]

L1 = tf.nn.conv2d(x, w1, strides=[1,1,1,1], padding='VALID') # stride도 4차원으로 / 가운데 2개가 실질적 stride, 양옆은 shape 맞추는 용도
# model.add(Conv2d(64, kernel_size=(2,2), input_shape=(28,28,1))) -> (27,27,1,64)

print(w1) # <tf.Variable 'w1:0' shape=(2, 2, 1, 64) dtype=float32_ref>
print(L1) # Tensor("Conv2D:0", shape=(?, 27, 27, 64), dtype=float32)








