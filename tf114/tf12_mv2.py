import tensorflow as tf
tf.compat.v1.set_random_seed(123)

x_data = [[73, 51, 65],                         # (5,3)
          [92, 98, 11],
          [89, 31, 33],
          [99, 33, 100],
          [17, 66, 79]]
y_data = [[152], [185], [180], [205], [142]]    # (5, 1)

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 3]) # 행렬형태부터는 shape를 명시해야함
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable()