import tensorflow as tf
from sklearn.metrics import accuracy_score, mean_squared_error
import numpy as np
import pandas as pd
tf.compat.v1.set_random_seed(1234)

# 1. 데이터
x_data = [[0,0], [0,1], [1,0], [1,1]] # (4, 2)
y_data = [[0],[1],[1],[0]] # (4, 1)

# 2. 모델
# input layer
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

# hidden layer
w1 = tf.compat.v1.Variable(tf.zeros([2,20]), name='weight')
b1 = tf.compat.v1.Variable(tf.zeros([20]), name='bias')
hidden_layer1 = tf.compat.v1.matmul(x, w1) + b1

w2 = tf.compat.v1.Variable(tf.zeros([20,30]), name='weight')
b2 = tf.compat.v1.Variable(tf.zeros([30]), name='bias')
hidden_layer2 = tf.compat.v1.sigmoid(tf.compat.v1.matmul(hidden_layer1, w2) + b2)

# output layer
w3 = tf.compat.v1.Variable(tf.zeros([30,1]), name='weight')
b3 = tf.compat.v1.Variable(tf.zeros([1]), name='bias')
hypothesis = tf.compat.v1.sigmoid(tf.compat.v1.matmul(hidden_layer2, w3) + b3)


# [실습]
# 3. 컴파일, 훈련
loss = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))
optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
train = optimizer.minimize(loss)

# 4. 평가, 예측
sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

epochs = 501
for step in range(epochs):
    _, hy_val, cost_val = sess.run([train,hypothesis,loss], feed_dict={x:x_data, y:y_data})
    if step%20 == 0:
        print(step, cost_val, hy_val)
        
print('최종: ', cost_val, hy_val)

predict = sess.run(tf.cast(hy_val>=0.5, dtype=tf.float32))
acc = accuracy_score(y_data, predict)
print('acc: ', acc)

mse = mean_squared_error(y_data, hy_val)
print('mae: ', mse)

sess.close()

