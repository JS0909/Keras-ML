import tensorflow as tf
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
tf.compat.v1.set_random_seed(123)

# 1. 데이터
datasets = load_breast_cancer()
x_data, y_data = datasets.data, datasets.target
print(x_data.shape, y_data.shape) # (569, 30) (569,)
print(type(x_data))
print(x_data.dtype, y_data.dtype)

# y_data = y_data.astype(np.float)
y_data = y_data.reshape(-1, 1)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.8, random_state=123, stratify=y_data)
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델 / sigmoid
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 30])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.compat.v1.zeros([30,5]), name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.zeros([5]), name='bias')
hidden = tf.compat.v1.nn.relu(tf.compat.v1.matmul(x, w) + b)

w = tf.compat.v1.Variable(tf.compat.v1.zeros([5,50]), name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.zeros([50]), name='bias')
hidden = tf.compat.v1.matmul(hidden, w) + b

w = tf.compat.v1.Variable(tf.compat.v1.zeros([50,40]), name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.zeros([40]), name='bias')
hidden = tf.compat.v1.nn.relu(tf.compat.v1.matmul(hidden, w) + b)

w = tf.compat.v1.Variable(tf.compat.v1.zeros([40,100]), name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.zeros([100]), name='bias')
hidden = tf.compat.v1.nn.relu(tf.compat.v1.matmul(hidden, w) + b)

w = tf.compat.v1.Variable(tf.compat.v1.zeros([100,20]), name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.zeros([20]), name='bias')
hidden = tf.compat.v1.nn.relu(tf.compat.v1.matmul(hidden, w) + b)

w = tf.compat.v1.Variable(tf.compat.v1.zeros([20,10]), name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.zeros([10]), name='bias')
hidden = tf.compat.v1.nn.relu(tf.compat.v1.matmul(hidden, w) + b)

w = tf.compat.v1.Variable(tf.compat.v1.zeros([10,1]), name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1]), name='bias')
hypothesis = tf.compat.v1.sigmoid(tf.compat.v1.matmul(hidden, w) + b)

# 3-1. 컴파일
loss = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis)) # binary_crossentropy

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-3)
optimizer = tf.train.AdamOptimizer(learning_rate=0.00000117)
train = optimizer.minimize(loss)

# 3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

epochs = 1001
for step in range(epochs):
    _, hy_val, cost_val, b_val = sess.run([train,hypothesis,loss,b], feed_dict={x:x_train, y:y_train})
    if step%20 == 0:
        print(step, cost_val, hy_val)
        
print('최종: ', cost_val, hy_val)

# 4. 평가, 예측
y_predict = sess.run(tf.cast(hy_val>=0.5, dtype=tf.float32))
acc = accuracy_score(y_train, y_predict)
print('acc: ', acc)

mae = mean_squared_error(y_train, hy_val)
print('mae: ', mae)

sess.close()

# acc:  0.6263736263736264
# mae:  0.24999260398557477