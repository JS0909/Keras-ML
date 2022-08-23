
import tensorflow as tf
print(tf.__version__)

# print('hello world')

hello = tf.constant('hello world')
print(hello)
# tf에는 constant, variable, placeholder 가 있다

# sess = tf.Session()
sess = tf.compat.v1.Session() # 이러면 워닝 안뜸
print(sess.run(hello))
# b'hello world' : 앞에 b는 바이너리형태라는 것

# 텐서플로우1은 출력할때 반드시 sess.run() 사용해야한다