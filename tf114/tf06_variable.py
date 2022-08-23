import tensorflow as tf
sess = tf.compat.v1.Session()

x = tf.Variable([2], dtype=tf.float32) # 변수는 초기값 지정을 해준다
y = tf.Variable([3], dtype=tf.float32)

# 변수를 선언한 후 초기화시키는 과정이 필요
# 이 명령어 이전에 지정한 변수들에 한해서 초기화함
init = tf.compat.v1.global_variables_initializer()
sess.run(init) # 꼭 실행시켜줘야됨

print(sess.run(x+y))