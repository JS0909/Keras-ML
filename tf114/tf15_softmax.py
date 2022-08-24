import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, mean_absolute_error
tf.set_random_seed(123)

x_data = [[1,2,1,1],  # x = (N, 4)   w = (4, 3)   y = (N, 3)   b = (1, 3)
          [2,1,3,2],
          [3,1,3,4],
          [4,1,5,5],
          [1,7,5,5],
          [1,2,5,6],
          [1,6,6,6],
          [1,7,6,7]]

y_data = [[0,0,1],
          [0,0,1],
          [0,0,1],
          [0,1,0],
          [0,1,0],
          [0,1,0],
          [1,0,0],
          [1,0,0]]

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 4])

w = tf.Variable(tf.random_normal([4, 3]), name='weight')
b = tf.Variable(tf.random_normal([1, 3]), name='bias')

y = tf.compat.v1.placeholder(tf.float32, shape=[None, 3])


# 2. 모델
hypothesis = tf.nn.softmax(tf.matmul(x, w) + b)
# model.add(Dense(3, activation='softmax', input_dim=4))

# 3-1. 컴파일
loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypothesis), axis=1))
# model.compile(loss='categorical_crossentropy')

train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

# 3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

epochs = 2001
for step in range(epochs):
    _, hy_val, cost_val, b_val = sess.run([train,hypothesis,loss,b], feed_dict={x:x_data, y:y_data})
    if step%20 == 0:
        print(step, cost_val, hy_val)
        
print('최종: ', cost_val, hy_val)

y_pred = sess.run(hypothesis, feed_dict={x:x_data, y:y_data})

acc = accuracy_score(np.argmax(y_data, axis=1), np.argmax(y_pred, axis=1))
print('acc: ', acc)

mae = mean_absolute_error(np.argmax(y_data, axis=1), np.argmax(y_pred, axis=1))
print('mae: ', mae)

# acc:  0.875
# mae:  0.125