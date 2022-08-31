import tensorflow as tf
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_covtype
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
tf.compat.v1.set_random_seed(123)

# 1. 데이터
data = fetch_covtype()
x, y = data.data, data.target
y = pd.get_dummies(y)

print(x.shape, y.shape) # (581012, 54) (581012, 7)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1234)
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 54])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 7])

w = tf.compat.v1.Variable(tf.compat.v1.zeros([54,10]), name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.zeros([10]), name='bias')
hidden = tf.nn.relu(tf.matmul(x, w) + b)

w = tf.compat.v1.Variable(tf.compat.v1.zeros([10,20]), name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.zeros([20]), name='bias')
hidden = tf.matmul(hidden, w) + b

w = tf.compat.v1.Variable(tf.compat.v1.zeros([20,15]), name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.zeros([15]), name='bias')
hidden = tf.matmul(hidden, w) + b

w = tf.compat.v1.Variable(tf.compat.v1.zeros([15,30]), name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.zeros([30]), name='bias')
hidden = tf.nn.relu(tf.matmul(hidden, w) + b)

w = tf.compat.v1.Variable(tf.compat.v1.zeros([30,20]), name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.zeros([20]), name='bias')
hidden = tf.nn.relu(tf.matmul(hidden, w) + b)

w = tf.compat.v1.Variable(tf.compat.v1.zeros([20,7]), name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.zeros([7]), name='bias')
hypothesis = tf.nn.softmax(tf.matmul(hidden, w) + b)

# 3-1. 컴파일
loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypothesis), axis=1))
# model.compile(loss='categorical_crossentropy')

train = tf.train.GradientDescentOptimizer(learning_rate=1e-7).minimize(loss)

# 3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

epochs = 50
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

# acc:  0.4878359422734353
# mae:  0.781993580200167