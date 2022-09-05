# [실습]
# DNN 구성

import tensorflow as tf
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error
import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical
tf.compat.v1.set_random_seed(123)

# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28* 28* 1).astype('float32')/255.
x_test = x_test.reshape(10000, 28* 28* 1).astype('float32')/255.

# 2. 모델
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 28*28])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 10])

w = tf.compat.v1.Variable(tf.compat.v1.zeros([28*28,64]), name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.zeros([64]), name='bias')
hidden = tf.compat.v1.nn.relu(tf.compat.v1.matmul(x, w) + b)

w = tf.compat.v1.Variable(tf.compat.v1.zeros([64,32]), name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.zeros([32]), name='bias')
hidden = tf.compat.v1.matmul(hidden, w) + b

w = tf.compat.v1.Variable(tf.compat.v1.zeros([32,10]), name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.zeros([10]), name='bias')
hypothesis = tf.compat.v1.nn.softmax(tf.compat.v1.matmul(hidden, w) + b)

# 3-1. 컴파일
loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypothesis), axis=1))
train = tf.train.AdadeltaOptimizer(learning_rate=1e-5).minimize(loss)

# 3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

epochs = 1001
for step in range(epochs):
    _, hy_val, cost_val, b_val = sess.run([train,hypothesis,loss,b], feed_dict={x:x_train, y:y_train})
    if step%20 == 0:
        print(step, cost_val, hy_val)
        
print('최종: ', cost_val, hy_val)

_, y_pred = sess.run([train,hypothesis], feed_dict={x:x_test, y:y_test})

acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
print('acc: ', acc)

mae = mean_absolute_error(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
print('mae: ', mae)

sess.close()

# acc:  0.1135
# mae:  3.6394