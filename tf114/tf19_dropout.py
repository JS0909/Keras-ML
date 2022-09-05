import tensorflow as tf

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w1 = tf.compat.v1.Variable(tf.random.normal([2, 30], name = 'weights1'))
b1 = tf.compat.v1.Variable(tf.random_normal([30], name = 'bias1'))

hidden_layer1 = tf.compat.v1.sigmoid(tf.matmul(x, w1) + b1)
# model.add(Dense(30, input_shape=(2,), activateion='sigmoid'))

dropout_layers = tf.compat.v1.nn.dropout(hidden_layer1, keep_prob=0.7) # 70% keep
dropout_layers = tf.compat.v1.nn.dropout(hidden_layer1, rate=0.3) # 30% drop

print(hidden_layer1) # Tensor("Sigmoid:0", shape=(?, 30), dtype=float32)
# 레이어 하나하나 볼 수 있음
print(dropout_layers) # Tensor("dropout/mul_1:0", shape=(?, 30), dtype=float32)

