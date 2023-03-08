import tensorflow as tf


a = tf.ones([50, 170, 4, 32])
b = tf.ones([170, 32, 1])

c = tf.matmul(a, b)

with tf.compat.v1.Session() as sess:
    m = sess.run(c)

    print(m.shape)
