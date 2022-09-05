import tensorflow as tf

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w1 = tf.compat.v1.Variable(tf.random.normal([2, 30], name = 'weights1'))
b1 = tf.compat.v1.Variable(tf.random_normal([30], name = 'bias1'))

hidden_layer1 = tf.compat.v1.sigmoid(tf.matmul(x, w1) + b1)
# model.add(Dense(30, input_shape=(2,), activateion='sigmoid'))





