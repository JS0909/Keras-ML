import tensorflow as tf
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
tf.compat.v1.set_random_seed(123)

# 1. 데이터
datasets = load_breast_cancer()
x_data, y_data = datasets.data, datasets.target
print(x_data.shape, y_data.shape) # (569, 30) (569,)
print(type(x_data))
print(x_data.dtype, y_data.dtype)

y_data = y_data.reshape(-1, 1)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.8, random_state=123, stratify=y_data)

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 30])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.compat.v1.zeros([30,1]), name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1]), name='bias')

# 2. 모델 / sigmoid
hypothesis = tf.compat.v1.sigmoid(tf.compat.v1.matmul(x, w) + b)

# 3-1. 컴파일
loss = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis)) # binary_crossentropy

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-3)
optimizer = tf.train.AdamOptimizer(learning_rate=0.00000117)
train = optimizer.minimize(loss)

# 3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

epochs = 5001
for step in range(epochs):
    _, hy_val, cost_val, b_val = sess.run([train,hypothesis,loss,b], feed_dict={x:x_train, y:y_train})
    if step%20 == 0:
        print(step, cost_val, hy_val)
        
print('최종: ', cost_val, hy_val)

# 4. 평가, 예측
predict = tf.cast(hypothesis>=0.5, dtype=tf.float32)
_, y_predict, _ = sess.run([hypothesis,predict,accuracy], feed_dict={x:x_test, y:y_test})
acc = accuracy_score(y_data, np.round(y_predict))
print('acc: ', acc)

mae = mean_squared_error(y_data, hy_val)
print('mae: ', mae)

sess.close()

