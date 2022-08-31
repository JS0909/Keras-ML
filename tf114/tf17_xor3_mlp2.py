
# [실습] 히든 레이어를 여러개로 구성해서 
# 1.0을 만들어 인공지능의 겨울 극복!!

from sklearn.model_selection import train_test_split
import tensorflow as tf
tf.compat.v1.set_random_seed(123)

#1. 데이터
x_data = [[0,0], [0,1], [1,0], [1,1]]   # (4, 2)
y_data = [[0], [1], [1], [0]]           # (4, 1)

# 2. 모델구성
# input layer
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

# hidden layer
w1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([2,20]), name='weight1')
b1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([20]), name='bias1')

hidden_layer1 = tf.compat.v1.matmul(x, w1) + b1

w2 = tf.compat.v1.Variable(tf.compat.v1.random_normal([20,30]), name='weight2')
b2 = tf.compat.v1.Variable(tf.compat.v1.random_normal([30]), name='bias2')

hidden_layer2 = tf.compat.v1.sigmoid(tf.compat.v1.matmul(hidden_layer1, w2) + b2)

w3 = tf.compat.v1.Variable(tf.compat.v1.random_normal([30,20]), name='weight3')
b3 = tf.compat.v1.Variable(tf.compat.v1.random_normal([20]), name='bias3')

hidden_layer3 = tf.compat.v1.sigmoid(tf.compat.v1.matmul(hidden_layer2, w3) + b3)

# output layer
w4 = tf.compat.v1.Variable(tf.compat.v1.random_normal([20,1]), name='weight4')
b4 = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), name='bias4')

hypothesis = tf.compat.v1.sigmoid(tf.compat.v1.matmul(hidden_layer3, w4) + b4)    # sigmoid 

#3-1. 컴파일
loss = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))  # binary_crossentropy
# optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-3)   
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3)   

train = optimizer.minimize(loss)

# 3-2. 훈련
sess = tf.compat.v1.Session() 
sess.run(tf.compat.v1.global_variables_initializer())     

epochs = 2001
for step in range(epochs):
    cost_val, hy_val, _ = sess.run([loss, hypothesis, train], 
                                    feed_dict={x:x_data, y:y_data})
    if step %20 == 0:
        print(step, 'loss : ', cost_val, '\n', hy_val)      
            
#4. 평가, 예측
y_predict = sess.run(tf.cast(hy_val>0.5, dtype=tf.float32))  
from sklearn.metrics import accuracy_score, mean_absolute_error
acc = accuracy_score(y_data, y_predict)
print('acc : ', acc)    
mae = mean_absolute_error(y_data, hy_val)
print('mae : ', mae)      
sess.close()

# acc :  1.0
# mae :  0.001247905194759369