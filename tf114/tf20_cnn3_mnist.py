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
w1 = tf.compat.v1.get_variable('w1', shape=[2, 2, 1, 50])
L1 = tf.nn.conv2d(x, w1, strides=[1,1,1,1], padding='SAME')
L1 = tf.nn.relu(L1)
L1_maxpool = tf.nn.max_pool2d(L1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
# 3x3커널이라면 stride도 3x3이어야 함, 안그럼 맥스풀링 의미가 달라진달까
# model.add(Conv2d(64, kernel_size=(2,2), input_shape=(28,28,1), activation='relu'))
print(L1_maxpool) # Tensor("MaxPool2d:0", shape=(?, 14, 14, 128), dtype=float32)

# Layer2
w2 = tf.compat.v1.get_variable('w2', shape=[3, 3, 50, 64])
L2 = tf.nn.conv2d(L1_maxpool, w2, strides=[1,1,1,1], padding='VALID')
L2 = tf.nn.selu(L2)
L2_maxpool = tf.compat.v1.nn.max_pool2d(L2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
print(L2) # Tensor("Conv2D_1:0", shape=(?, 12, 12, 64), dtype=float32)
print(L2_maxpool) # Tensor("MaxPool2d_1:0", shape=(?, 6, 6, 64), dtype=float32)

# Layer3
w3 = tf.compat.v1.get_variable('w3', shape=[3, 3, 64, 32])
L3 = tf.nn.conv2d(L2_maxpool, w3, strides=[1,1,1,1], padding='VALID')
L3 = tf.nn.elu(L3)
print(L3) # Tensor("Elu:0", shape=(?, 4, 4, 32), dtype=float32)

# Flatten
L_flat = tf.reshape(L3, [-1, 4*4*32])
print('flatten: ', L_flat) # Tensor("Reshape:0", shape=(?, 512), dtype=float32)

# Layer4 DNN
w4 = tf.compat.v1.get_variable('w4', shape=[4*4*32, 50])
                    #  initializer=tf.contrib.layers.xavier_initializer()) # 가중치 초기화, 제한
b4 = tf.Variable(tf.compat.v1.random_normal([50]), name='b4')
L4 = tf.nn.selu(tf.matmul(L_flat, w4) + b4)
L4 = tf.nn.dropout(L4, rate=0.3)

# Layer5 DNN
w5 = tf.compat.v1.get_variable('w5', shape=[50, 10],)
                    #  initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.compat.v1.random_normal([10]), name='b5')
L5 = tf.matmul(L4, w5) + b5
hypothesis = tf.nn.softmax(L5)
print(hypothesis) # Tensor("Softmax:0", shape=(?, 10), dtype=float32)

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
    
    prediction = tf.equal(tf.compat.v1.arg_max(hypothesis, 1), tf.argmax(y, 1))
    acc = tf.reduce_mean(tf.cast(prediction, tf.float32))
    print('epoch: ', '%04d'%(epoch + 1), 'loss: {:.9f}'.format(avg_loss), 'ACC: ', acc)
    
print('훈련 완료')

prediction = tf.equal(tf.compat.v1.arg_max(hypothesis, 1), tf.argmax(y, 1)) # equal(a, b) : a=b 면 1 / a!=b면 0
acc = tf.reduce_mean(tf.cast(prediction, tf.float32))
print('ACC: ', sess.run(acc, feed_dict={x:x_test, y:y_test}))

sess.close()
