import tensorflow as tf

#The value returned by the constructor representes the output of
#the constant op.
a = tf.constant(2)
b = tf.constant(3)

with tf.Session() as sess:
    print('a=' + str(sess.run(a)))
    print('b=' + str(sess.run(b)))
    print('a + b = ' + str(sess.run(a + b)))

#a=2
#b=3
#a + b = 5


