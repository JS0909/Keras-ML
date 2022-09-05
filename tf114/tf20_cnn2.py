import tensorflow as tf
import numpy as np
tf.compat.v1.set_random_seed(123)

# 1. data
x_train = np.array([[[[1], [2], [3]],
                     [[4], [5], [6]],
                     [[7], [8], [9]]]])
print(x_train.shape) # (1, 3, 3, 1)

x = tf.compat.v1.placeholder(tf.float32, [None, 3, 3, 1])

w = tf.compat.v1.constant([[[[1.]], [[1.]]],
                           [[[1.]], [[1.]]]])

print(w) # Tensor("Const:0", shape=(2, 2, 1, 1), dtype=float32)

# L1 = tf.nn.conv2d(x, w, strides=(1,1,1,1), padding='VALID')
# print(L1) # Tensor("Conv2D:0", shape=(?, 2, 2, 1), dtype=float32) / 커널사이즈때문에 하나가 줄었음
# L1 = tf.nn.conv2d(x, w, strides=(1,1,1,1), padding='SAME')
L1 = tf.nn.conv2d(x, w, strides=(1,2,2,1), padding='SAME')
print(L1)


sess = tf.compat.v1.Session()
output = sess.run(L1, feed_dict={x:x_train})
print(output)
print(output.shape)



