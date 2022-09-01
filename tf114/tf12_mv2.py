import tensorflow as tf
from sklearn.metrics import r2_score, mean_absolute_error
tf.compat.v1.set_random_seed(66)

x_data = [[73, 51, 65],     # (5,3)
          [92, 98, 11],
          [89, 31, 33],
          [99, 33, 100],
          [17, 66, 79]]

y_data = [[152],            # (5, 1)
          [185], 
          [180], 
          [205], 
          [142]]    

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 3]) # 행렬형태부터는 shape를 명시해야함
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([3,1]), name='weight')
# 곱해질 w의 행값은 x_data의 열값과 같아야하고, w의 열값은 y_data의 열값과 같아야함
# (5, 3) * (3, 1) = (5, 1)
b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), name='bias')

# 2. 모델
hypothesis = tf.compat.v1.matmul(x, w) + b

# 3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y_data))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(loss)

# 3-2. 훈련
with tf.compat.v1.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    epochs = 101
    for step in range(epochs):
        _, hy_val, cost_val = sess.run([train,hypothesis,loss], feed_dict={x:x_data, y:y_data})
        if step%20 == 0:
            print(step, cost_val, hy_val)
            
    print('최종: ', cost_val, hy_val)


    r2 = r2_score(y_data, hy_val)
    print('r2: ', r2)

    mae = mean_absolute_error(y_data, hy_val)
    print('mae: ', mae)

# r2:  0.41929962984603475
# mae:  14.5033447265625