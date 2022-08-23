import tensorflow as tf
print(tf.__version__)
print(tf.executing_eagerly()) # 텐서플로2는 True로 되어 있음 / 즉시실행모드임

# 즉시실행모드 끄기, 즉 텐서플로1로 실행하겠다
tf.compat.v1.disable_eager_execution()

print(tf.executing_eagerly())

hello = tf. constant('hello world')

sess = tf.compat.v1.Session()
print(sess.run(hello))