# [실습]
# DNN 구성

import tensorflow as tf
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error
import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical
tf.compat.v1.set_random_seed(123)
tf.compat.v1.disable_eager_execution()

# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255.

# 2. 모델
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 28, 28, 1])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 10])

# Layer1
w1 = tf.compat.v1.get_variable('w1', shape=[3, 3, 1, 64])
L1 = tf.nn.conv2d(x, w1, strides=[1,1,1,1], padding='SAME')
L1 = tf.nn.relu(L1)
L1_maxpool = tf.nn.max_pool2d(L1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

# Layer2
w2 = tf.compat.v1.get_variable('w2', shape=[2, 2, 64, 10])
L2 = tf.nn.conv2d(L1_maxpool, w2, strides=[1,1,1,1], padding='VALID')
L2 = tf.nn.relu(L2)

# Layer3
w3 = tf.compat.v1.get_variable('w3', shape=[2, 2, 10, 5])
L3 = tf.nn.conv2d(L2, w3, strides=[1,1,1,1], padding='SAME')
L3 = tf.nn.relu(L3)

# Layer4
w4 = tf.compat.v1.get_variable('w4', shape=[2, 2, 5, 4])
L4 = tf.nn.conv2d(L3, w4, strides=[1,1,1,1], padding='VALID')
L4 = tf.nn.relu(L4)
# Tensor("Relu_3:0", shape=(None, 12, 12, 4), dtype=float32)

# Flatten
L_flat = tf.reshape(L4, [-1, 12*12*4])

# Layer5 DNN
w5 = tf.compat.v1.get_variable('w5', shape=[12*12*4, 32])
b5 = tf.Variable(tf.compat.v1.random_normal([32]), name='b5')
L5 = tf.nn.relu(tf.matmul(L_flat, w5) + b5)

# Layer6 DNN
w6 = tf.compat.v1.get_variable('w6', shape=[32, 32])
b6 = tf.Variable(tf.compat.v1.random_normal([32]), name='b6')
L6 = tf.nn.relu(tf.matmul(L5, w6) + b6)

# Layer7 DNN
w7 = tf.compat.v1.get_variable('w7', shape=[32, 10])
b7 = tf.Variable(tf.compat.v1.random_normal([10]), name='b7')
hypothesis = tf.nn.relu(tf.matmul(L6, w7) + b7)



# 3-1. 컴파일
# loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypothesis), axis=1))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=y))
optimizer = tf.compat.v1.train.AdadeltaOptimizer(learning_rate=1e-3).minimize(loss)

# 3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 30
batch_size = 300
total_batch = int(len(x_train)/batch_size) # 1에포 당 600번 돈다

for epoch in range(epochs):
    avg_loss = 0
    for i in range(total_batch): # 총 600번 돈다
        start = i * batch_size   # i=0일 때 0
        end = start + batch_size # i=0일 때 100
        batch_x, batch_y = x_train[start:end], y_train[start:end] # i=0일 때 0~100
        
        feed_dict = {x:batch_x, y:batch_y}
        
        batch_loss, _ = sess.run([loss, optimizer], feed_dict=feed_dict)
        
        avg_loss += batch_loss / total_batch
    
    # prediction = tf.equal(tf.compat.v1.arg_max(hypothesis, 1), tf.argmax(y, 1))
    # acc = tf.reduce_mean(tf.cast(prediction, tf.float32))
    print('epoch: ', '%04d'%(epoch + 1), 'loss: {:.9f}'.format(avg_loss), 'ACC: ',)
    
print('훈련 완료')

prediction = tf.equal(tf.compat.v1.arg_max(hypothesis, 1), tf.argmax(y, 1)) # equal(a, b) : a=b 면 1 / a!=b면 0
acc = tf.reduce_mean(tf.cast(prediction, tf.float32))
print('ACC: ', sess.run(acc, feed_dict={x:x_test, y:y_test}))

sess.close()
