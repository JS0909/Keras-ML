import tensorflow as tf
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
import numpy as np
import pandas as pd
tf.compat.v1.set_random_seed(123)

# 1. 데이터
data = load_wine()
x, y = data.data, data.target
y = pd.get_dummies(y)

print(x.shape, y.shape) # (178, 13) (178, 3)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1234)


# 2. 모델
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 13])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 3])

w = tf.compat.v1.Variable(tf.compat.v1.zeros([13,80]), name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.zeros([80]), name='bias')
hidden = tf.nn.relu(tf.matmul(x, w) + b)

w = tf.compat.v1.Variable(tf.compat.v1.zeros([80,100]), name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.zeros([100]), name='bias')
hidden = tf.matmul(hidden, w) + b

w = tf.compat.v1.Variable(tf.compat.v1.zeros([100,90]), name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.zeros([90]), name='bias')
hidden = tf.matmul(hidden, w) + b

w = tf.compat.v1.Variable(tf.compat.v1.zeros([90,70]), name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.zeros([70]), name='bias')
hidden = tf.nn.relu(tf.matmul(hidden, w) + b)

w = tf.compat.v1.Variable(tf.compat.v1.zeros([70,50]), name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.zeros([50]), name='bias')
hidden = tf.nn.relu(tf.matmul(hidden, w) + b)

w = tf.compat.v1.Variable(tf.compat.v1.zeros([50,3]), name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.zeros([3]), name='bias')
hypothesis = tf.nn.softmax(tf.matmul(hidden, w) + b)


# 3-1. 컴파일
loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypothesis), axis=1))

train = tf.train.GradientDescentOptimizer(learning_rate=1e-5).minimize(loss)

# 3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

epochs = 2001
for step in range(epochs):
    _, hy_val, cost_val, b_val = sess.run([train,hypothesis,loss,b], feed_dict={x:x_train, y:y_train})
    if step%20 == 0:
        print(step, cost_val, hy_val)
        
print('최종: ', cost_val, hy_val)

_, y_pred = sess.run([train,hypothesis], feed_dict={x:x_test, y:y_test})

y_test = y_test.values

acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
print('acc: ', acc)

mae = mean_absolute_error(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
print('mae: ', mae)

# acc:  0.4722222222222222
# mae:  0.5277777777777778