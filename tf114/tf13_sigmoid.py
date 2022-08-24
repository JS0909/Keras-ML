import tensorflow as tf
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error
import numpy as np
tf.compat.v1.set_random_seed(123)

# 1. 데이터
x_data = [[1,2],[2,3],[3,1],[4,3],[5,3], [6,2]] # (6, 2)
y_data = [[0], [0], [0], [1], [1], [1]]         # (6, 1)

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 2]) # 행렬형태부터는 shape를 명시해야함
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([2,1]), name='weight')
# 곱해질 w의 행값은 x_data의 열값과 같아야하고, w의 열값은 y_data의 열값과 같아야함
# (5, 3) * (3, 1) = (5, 1)
b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), name='bias')

# 2. 모델 / sigmoid
hypothesis = tf.compat.v1.sigmoid(tf.compat.v1.matmul(x, w) + b)
# model.add(Dense(1, activation='sigmoid', input_dim=2))

# 3-1. 컴파일
# loss = tf.reduce_mean(tf.square(hypothesis - y_data)) # mse
loss = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis)) # binary_crossentropy
# model.compile(loss='binary_crossentropy')

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-1)
train = optimizer.minimize(loss)

# 3-2. 훈련
with tf.compat.v1.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    epochs = 1001
    for step in range(epochs):
        _, hy_val, cost_val = sess.run([train,hypothesis,loss], feed_dict={x:x_data, y:y_data})
        if step%20 == 0:
            print(step, cost_val, hy_val)
            
    print('최종: ', cost_val, hy_val)

    y_predict = sess.run(tf.cast(hy_val>=0.5, dtype=tf.float32))
    # tf.cast: 해당 조건이 참이면 1, 아니면 0 반환
    # tf형이기 때문에 cast도 run 쳐줘야됨

    acc = accuracy_score(y_data, y_predict)
    print('acc: ', acc)

    mae = mean_absolute_error(y_data, hy_val)
    print('mae: ', mae)

