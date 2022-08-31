import tensorflow as tf
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
import numpy as np
tf.compat.v1.set_random_seed(123)

# 1. 데이터
data = fetch_california_housing()
x, y = data.data, data.target
y = y.reshape(-1, 1)
print(x.shape, y.shape) # (20640, 8) (20640, 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1234)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# 2. 모델
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 8])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.compat.v1.zeros([8,20]), name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.zeros([20]), name='bias')
hidden = tf.compat.v1.matmul(x, w) + b

w = tf.compat.v1.Variable(tf.compat.v1.zeros([20,10]), name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.zeros([10]), name='bias')
hidden = tf.compat.v1.matmul(hidden, w) + b

w = tf.compat.v1.Variable(tf.compat.v1.zeros([10,70]), name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.zeros([70]), name='bias')
hidden = tf.compat.v1.matmul(hidden, w) + b

w = tf.compat.v1.Variable(tf.compat.v1.zeros([70,50]), name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.zeros([50]), name='bias')
hidden = tf.compat.v1.matmul(hidden, w) + b

w = tf.compat.v1.Variable(tf.compat.v1.zeros([50,10]), name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.zeros([10]), name='bias')
hidden = tf.compat.v1.matmul(hidden, w) + b

w = tf.compat.v1.Variable(tf.compat.v1.zeros([10,1]), name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1]), name='bias')
hypothesis = tf.compat.v1.matmul(hidden, w) + b

# 3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis-y))

optimizer = tf.train.AdamOptimizer(learning_rate=1e-7)
train = optimizer.minimize(loss)

# 3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

epochs = 2001
for step in range(epochs):
    _, hy_val, cost_val, b_val = sess.run([train,hypothesis,loss,b], feed_dict={x:x_train, y:y_train})
    if step%20 == 0:
        print(step, cost_val, hy_val)
        
print('최종: ', cost_val, hy_val)

y_pred = sess.run(hypothesis, feed_dict={x:x_test, y:y_test})

r2 = r2_score(y_test, y_pred)
print('r2: ', r2)

mae = mean_absolute_error(y_test, y_pred)
print('mae: ', mae)

# r2:  -3.187749156450473
# mae:  2.076638602300476