# y = wx + b
import tensorflow as tf
tf.set_random_seed(123)

# 1. 데이터
x = [1, 2, 3, 4, 5]
y = [1, 2, 3, 4, 5]

W = tf.Variable(33, dtype=tf.float32)
b = tf.Variable(11, dtype=tf.float32)

# 2. 모델구성
hypothesis = x * W + b # y = wx + b
# 행렬연산이기 때문에 x와 W 순서가 중요하다, 즉 인풋값(x)에 웨이트(w)를 곱한다

# 3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y)) # mse  /  원래 y와 h=wx+b의 거리를 구해서 n빵

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.03) # 러닝레이트는 내려가는 간격을 말함
train = optimizer.minimize(loss) # 정의된 optimizer방식으로 뽑은 loss들 중 최소값을 리턴하는 방식으로 훈련하겠다
# model.compile(loss='mse', optimizer='sgd')  /  텐서2에서는 이렇게 쓰면 됨

# 3-2. 훈련
with tf.compat.v1.Session() as sess: # 이런식으로 쓰면 close()를 쓰지 않아도 된다
    sess.run(tf.global_variables_initializer())

    epochs = 2556
    for step in range(epochs):
        sess.run(train)
        if step%50 == 0:
            print(step, sess.run(loss), sess.run(W), sess.run(b))

    print('최종: ', step, sess.run(loss), sess.run(W), sess.run(b))
