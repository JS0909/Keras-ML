import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import numpy as np
import pandas as pd

# pandas의 y라벨의 종류 확인 train_set.columns.values
# numpy에서는 np.unique(y, return_counts=True)

# 1. 데이터
path = 'D:/study_data/_data/kaggle_titanic/'
train_set = pd.read_csv(path+'train.csv')
test_set = pd.read_csv(path+'test.csv')

print(train_set.describe())
print(train_set.info())
print(train_set.isnull())
print(train_set.isnull().sum())
print(train_set.shape) # (10886, 12)
print(train_set.columns.values) # ['PassengerId' 'Survived' 'Pclass' 'Name' 'Sex' 'Age' 'SibSp' 'Parch' 'Ticket' 'Fare' 'Cabin' 'Embarked']

train_set = train_set.drop(columns='Cabin', axis=1)
train_set['Age'].fillna(train_set['Age'].mean(), inplace=True)
print(train_set['Embarked'].mode())
train_set['Embarked'].fillna(train_set['Embarked'].mode()[0], inplace=True)
train_set.replace({'Sex':{'male':0,'female':1}, 'Embarked':{'S':0,'C':1,'Q':2}}, inplace=True)

# train_set 불러올 때와 마찬가지로 전처리시켜야 model.predict에 넣어서 y값 구하기가 가능함-----------
print(test_set.isnull().sum())
test_set = test_set.drop(columns='Cabin', axis=1)
test_set['Age'].fillna(test_set['Age'].mean(), inplace=True)
test_set['Fare'].fillna(test_set['Fare'].mean(), inplace=True)
print(test_set['Embarked'].mode())
test_set['Embarked'].fillna(test_set['Embarked'].mode()[0], inplace=True)
test_set.replace({'Sex':{'male':0,'female':1}, 'Embarked':{'S':0,'C':1,'Q':2}}, inplace=True)
test_set = test_set.drop(columns = ['PassengerId','Name','Ticket'],axis=1)
#---------------------------------------------------------------------------------------------------

y = train_set['Survived']
x = train_set.drop(columns = ['PassengerId','Name','Ticket','Survived'],axis=1) 
y = np.array(y).reshape(-1, 1) # 벡터로 표시되어 있는 y데이터를 행렬로 전환

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8,shuffle=True, random_state=9)
print(x_train.shape, x_test.shape) # (712, 7) (179, 7)

# 2. 모델 / sigmoid
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 7])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.compat.v1.zeros([7,80]), name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.zeros([80]), name='bias')
hidden = tf.compat.v1.nn.relu(tf.compat.v1.matmul(x, w) + b)

w = tf.compat.v1.Variable(tf.compat.v1.zeros([80,100]), name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.zeros([100]), name='bias')
hidden = tf.compat.v1.matmul(hidden, w) + b

w = tf.compat.v1.Variable(tf.compat.v1.zeros([100,90]), name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.zeros([90]), name='bias')
hidden = tf.compat.v1.nn.relu(tf.compat.v1.matmul(hidden, w) + b)

w = tf.compat.v1.Variable(tf.compat.v1.zeros([90,70]), name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.zeros([70]), name='bias')
hidden = tf.compat.v1.nn.relu(tf.compat.v1.matmul(hidden, w) + b)

w = tf.compat.v1.Variable(tf.compat.v1.zeros([70,50]), name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.zeros([50]), name='bias')
hidden = tf.compat.v1.nn.relu(tf.compat.v1.matmul(hidden, w) + b)

w = tf.compat.v1.Variable(tf.compat.v1.zeros([50,1]), name='weight')
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

# acc:  0.6306179775280899
# mae:  0.24992371032655764