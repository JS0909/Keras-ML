import tensorflow as tf
tf.compat.v1.set_random_seed(123)

# [실습]
# 1. Session() // sess.run(변수)==================================================================
# 1. 데이터
x_train_data = [1,2,3]
y_train_data = [3,5,7]

x_train = tf.placeholder(tf.float32, shape=[None]) # shape 모르면 None하면 자동으로 잡음
y_train = tf.placeholder(tf.float32, shape=[None])

W = tf.Variable(tf.random_normal([1]), dtype=tf.float32) # []의 숫자는 랜덤숫자의 개수
b = tf.Variable(tf.random_normal([1]), dtype=tf.float32)
# 랜덤값을 줄 때 normal: 표준분포에 따른 랜덤 값 생성
# uniform: 0~1 사이의 균등확률분포 값을 생성, 모든 확률이 균일한 분포

sess = tf.compat.v1.Session()
# sess.run(tf.global_variables_initializer())

# 2. 모델구성
hypothesis = x_train * W + b

# 3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y_train))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.172)
train = optimizer.minimize(loss)

# 3-2. 훈련
with tf.compat.v1.Session() as sess:
    sess.run(tf.global_variables_initializer())
    W2 = sess.run(W)
    b2 = sess.run(b)

    epochs = 101
    for step in range(epochs):
        # sess.run(train)
        _, loss_val, W_val, b_val = sess.run([train,loss,W,b], feed_dict={x_train:x_train_data, y_train:y_train_data})
        if step%20 == 0:
            print(step, loss_val, W_val, b_val)

    print('최종스텝:', step, loss_val, '\nW: ', W_val, 'bias: ',b_val)

#================================= <Predict> ==============================================
    x_test_data = [6, 7, 8]
    x_test = tf.compat.v1.placeholder(tf.float32, shape=[None])

    y_predict = x_test * W_val + b_val # y_predict = model.predict(x_test)

    print('[6,7,8] 예측: ', sess.run(y_predict, feed_dict={x_test:x_test_data}))
    


# 2. Session() // 변수.eval(session=sess)==================================================================
# 1. 데이터
x_train_data = [1,2,3]
y_train_data = [3,5,7]

x_train = tf.placeholder(tf.float32, shape=[None]) # shape 모르면 None하면 자동으로 잡음
y_train = tf.placeholder(tf.float32, shape=[None])

W = tf.Variable(tf.random_normal([1]), dtype=tf.float32) # []의 숫자는 랜덤숫자의 개수
b = tf.Variable(tf.random_normal([1]), dtype=tf.float32)
# 랜덤값을 줄 때 normal: 표준분포에 따른 랜덤 값 생성
# uniform: 0~1 사이의 균등확률분포 값을 생성, 모든 확률이 균일한 분포

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())
W3 = W.eval(session=sess)
b3 = b.eval(session=sess)

# 2. 모델구성
hypothesis = x_train * W + b

# 3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y_train))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.172)
train = optimizer.minimize(loss)

# 3-2. 훈련
with tf.compat.v1.Session() as sess:
    sess.run(tf.global_variables_initializer())

    epochs = 101
    for step in range(epochs):
        # sess.run(train)
        _, loss_val, W_val, b_val = sess.run([train,loss,W,b], feed_dict={x_train:x_train_data, y_train:y_train_data})
        if step%20 == 0:
            print(step, loss_val, W_val, b_val)

    print('최종스텝:', step, loss_val, '\nW: ', W_val, 'bias: ',b_val)

#================================= <Predict> ==============================================
    x_test_data = [6, 7, 8]
    x_test = tf.compat.v1.placeholder(tf.float32, shape=[None])

    y_predict = x_test * W_val + b_val # y_predict = model.predict(x_test)

    print('[6,7,8] 예측: ', y_predict.eval(session=sess, feed_dict={x_test:x_test_data}))



# 3. InteractiveSession() // 변수.eval()==================================================================
# 1. 데이터
x_train_data = [1,2,3]
y_train_data = [3,5,7]

x_train = tf.placeholder(tf.float32, shape=[None]) # shape 모르면 None하면 자동으로 잡음
y_train = tf.placeholder(tf.float32, shape=[None])

W = tf.Variable(tf.random_normal([1]), dtype=tf.float32) # []의 숫자는 랜덤숫자의 개수
b = tf.Variable(tf.random_normal([1]), dtype=tf.float32)
# 랜덤값을 줄 때 normal: 표준분포에 따른 랜덤 값 생성
# uniform: 0~1 사이의 균등확률분포 값을 생성, 모든 확률이 균일한 분포

sess = tf.compat.v1.InteractiveSession()
sess.run(tf.global_variables_initializer())
W4 = W.eval()
b4 = b.eval()
sess.close()

# 2. 모델구성
hypothesis = x_train * W + b

# 3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y_train))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.172)
train = optimizer.minimize(loss)

# 3-2. 훈련
sess = tf.compat.v1.InteractiveSession()


sess.run(tf.global_variables_initializer())

epochs = 101
for step in range(epochs):
    # sess.run(train)
    _, loss_val, W_val, b_val = sess.run([train,loss,W,b], feed_dict={x_train:x_train_data, y_train:y_train_data})
    if step%20 == 0:
        print(step, loss_val, W_val, b_val)

print('최종스텝:', step, loss_val, '\nW: ', W_val, 'bias: ',b_val)

#================================= <Predict> ==============================================
x_test_data = [6, 7, 8]
x_test = tf.compat.v1.placeholder(tf.float32, shape=[None])

y_predict = x_test * W_val + b_val # y_predict = model.predict(x_test)

print('[6,7,8] 예측: ', sess.run(y_predict, feed_dict={x_test:x_test_data}))

