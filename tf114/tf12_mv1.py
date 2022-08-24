import tensorflow as tf
from sklearn.metrics import r2_score, mean_absolute_error
tf.compat.v1.set_random_seed(123)

# 1. 데이터
x1_data = [73.,93.,89.,96.,73.]
x2_data = [80.,88.,91.,98.,66.]
x3_data = [75.,93.,90.,100.,70.]
y_data = [152.,185.,180.,196.,142.]

x1 = tf.compat.v1.placeholder(tf.float32)
x2 = tf.compat.v1.placeholder(tf.float32, shape=[None])
x3 = tf.compat.v1.placeholder(tf.float32, shape=[None])

w1 =  tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), name='weight1')
w2 =  tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), name='weight2')
w3 =  tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), name='weight3')

b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), name='bias')

# 2. 모델
hypothesis = x1*w1 + x2*w2 + x3*w3 +b

# 3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y_data))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=4e-5)
train = optimizer.minimize(loss)

# 3-2. 훈련
with tf.compat.v1.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    epochs = 101
    for step in range(epochs):
        _, hy_val, cost_val = sess.run([train,hypothesis,loss], feed_dict={x1:x1_data, x2:x2_data, x3:x3_data})
        if step%20 == 0:
            print(step, cost_val, hy_val)
            
    print('최종: ', cost_val, hy_val)


    r2 = r2_score(y_data, hy_val)
    print('r2: ', r2)

    mae = mean_absolute_error(y_data, hy_val)
    print('mae: ', mae)