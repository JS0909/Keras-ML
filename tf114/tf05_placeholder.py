# variable은 인풋과 연산에서 사용 하지만 placeholder는 인풋에만 사용한다

import tensorflow as tf
print(tf.__version__)
print(tf.executing_eagerly()) # True

# 즉시실행모드 끄기
tf.compat.v1.disable_eager_execution()
print(tf.executing_eagerly()) # False

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)

sess = tf.compat.v1.Session()

a = tf.compat.v1.placeholder(tf.float32) # float32를 먹는 공간을 만든다
b = tf.compat.v1.placeholder(tf.float32)

add_node = a + b
print(sess.run(add_node, feed_dict={a:3, b:4.5})) # feed로 딕셔너리형태로 먹이를 던져줌
print(sess.run(add_node, feed_dict={a:[1,3], b:[2,4]})) # 다차원 행렬로 던져줄 수도 있다
print(add_node) # Tensor("add_1:0", dtype=float32)

add_and_triple = add_node * 3
print(sess.run(add_and_triple, feed_dict={a:3, b:4.5})) # feed로 딕셔너리형태로 먹이를 던져줌
print(sess.run(add_and_triple, feed_dict={a:[1,3], b:[2,4]})) # 다차원 행렬로 던져줄 수도 있다
print(add_and_triple) # Tensor("mul:0", dtype=float32)



