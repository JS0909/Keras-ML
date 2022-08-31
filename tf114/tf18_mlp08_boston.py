import tensorflow as tf
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
import numpy as np
from sklearn.preprocessing import StandardScaler
tf.compat.v1.set_random_seed(123)

# 1. 데이터
data = load_boston()
x, y = data.data, data.target
y = y.reshape(-1, 1)
print(x.shape, y.shape) # (506, 13) (506, 1)

x_train, x_test, y_train, y_test =  train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 13])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

# 2. 모델
w = tf.compat.v1.Variable(tf.compat.v1.zeros([13,100]), name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.zeros([100]), name='bias')
hidden = tf.compat.v1.matmul(x, w) + b

w = tf.compat.v1.Variable(tf.compat.v1.zeros([100,50]), name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.zeros([50]), name='bias')
hidden = tf.nn.relu(tf.compat.v1.matmul(hidden, w) + b)

w = tf.compat.v1.Variable(tf.compat.v1.zeros([50,16]), name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.zeros([16]), name='bias')
hidden = tf.nn.relu(tf.compat.v1.matmul(hidden, w) + b)

w = tf.compat.v1.Variable(tf.compat.v1.zeros([16,8]), name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.zeros([8]), name='bias')
hidden = tf.compat.v1.matmul(hidden, w) + b

w = tf.compat.v1.Variable(tf.compat.v1.zeros([8,1]), name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1]), name='bias')
hypothesis = tf.compat.v1.matmul(hidden, w) + b

# 3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis-y))

optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
train = optimizer.minimize(loss)

# 3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

epochs = 3001
for step in range(epochs):
    _, hy_val, cost_val, b_val = sess.run([train,hypothesis,loss,b], feed_dict={x:x_train, y:y_train})
    # if step%20 == 0:
    #     print(step, cost_val, hy_val)
        
print('최종: ', cost_val, hy_val)

y_pred = sess.run(hypothesis, feed_dict={x:x_test, y:y_test})

r2 = r2_score(y_test, y_pred)
print('r2: ', r2)

mae = mean_absolute_error(y_test, y_pred)
print('mae: ', mae)

# r2:  -4.677203188122392
# mae:  19.772060614941164