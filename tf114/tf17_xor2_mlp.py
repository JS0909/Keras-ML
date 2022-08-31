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
w1 = tf.compat.v1.Variable(tf.random_normal([2,20]), name='weight')
b1 = tf.compat.v1.Variable(tf.random_normal([20]), name='bias')
hidden_layer1 = tf.compat.v1.matmul(x, w1) + b1


# output layer
w2 = tf.compat.v1.Variable(tf.random_normal([20,1]), name='weight')
b2 = tf.compat.v1.Variable(tf.random_normal([1]), name='bias')
hypothesis = tf.compat.v1.sigmoid(tf.compat.v1.matmul(hidden_layer1, w2) + b2)


# [실습]
# 3. 컴파일, 훈련
loss = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-2)
train = optimizer.minimize(loss)

# 4. 평가, 예측
sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

epochs = 5001
for step in range(epochs):
    _, hy_val, cost_val, b_val = sess.run([train,hypothesis,loss,b1], feed_dict={x:x_data, y:y_data})
    if step%20 == 0:
        print(step, cost_val, hy_val)
        
print('최종: ', cost_val, hy_val)

predict = sess.run(tf.cast(hy_val>=0.5, dtype=tf.float32))
acc = accuracy_score(y_data, predict)
print('acc: ', acc)

mse = mean_squared_error(y_data, hy_val)
print('mae: ', mse)

sess.close()

