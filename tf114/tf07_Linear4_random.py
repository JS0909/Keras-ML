# 웨이트 변수를 텐서플로2처럼 랜덤하게 넣기

import tensorflow as tf
tf.set_random_seed(123)

# 1. 데이터
x = [1, 2, 3, 4, 5]
y = [1, 2, 3, 4, 5]

W = tf.Variable(tf.random_normal([1]), dtype=tf.float32) # []의 숫자는 랜덤숫자의 개수
b = tf.Variable(tf.random_normal([1]), dtype=tf.float32)
# 랜덤값을 줄 때 normal: 표준분포에 따른 랜덤 값 생성
# uniform: 0~1 사이의 균등확률분포 값을 생성, 모든 확률이 균일한 분포


sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(W))
'''
# 2. 모델구성
hypothesis = x * W + b

# 3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.03)
train = optimizer.minimize(loss)

# 3-2. 훈련
with tf.compat.v1.Session() as sess:
    sess.run(tf.global_variables_initializer())

    epochs = 2556
    for step in range(epochs):
        sess.run(train)
        if step%50 == 0:
            print(step, sess.run(loss), sess.run(W), sess.run(b))

    print('최종: ', step, sess.run(loss), sess.run(W), sess.run(b))
'''