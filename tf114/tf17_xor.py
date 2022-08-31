import tensorflow as tf
from sklearn.metrics import accuracy_score, mean_squared_error
import numpy as np
import pandas as pd
tf.compat.v1.set_random_seed(123)

# 1. 데이터
x_data = [[0,0], [0,1], [1,0], [1,1]] # (4, 2)
y_data = [[0],[1],[1],[0]] # (4, 1)

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.random_normal([2,1]))
b = tf.compat.v1.Variable(tf.random_normal([1]))

# 2. 모델
hypothesis = tf.compat.v1.sigmoid(tf.compat.v1.matmul(x, w) + b)


# [실습]
# 3. 컴파일, 훈련
loss = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-1)
train = optimizer.minimize(loss)

# 4. 평가, 예측
sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

epochs = 100
for step in range(epochs):
    _, hy_val, cost_val, b_val = sess.run([train,hypothesis,loss,b], feed_dict={x:x_data, y:y_data})
    if step%20 == 0:
        print(step, cost_val, hy_val)
        
print('최종: ', cost_val, hy_val)

predict = sess.run(tf.cast(hy_val>=0.5, dtype=tf.float32))
acc = accuracy_score(y_data, predict)
print('acc: ', acc)

mse = mean_squared_error(y_data, hy_val)
print('mae: ', mse)

sess.close()

