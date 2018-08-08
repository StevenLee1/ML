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



#Basic Operations with variable as graph input
#tf Graph input
a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)
#define operations
add = tf.add(a, b)
multiply = tf.multiply(a, b)
#launch the default graph
with tf.Session() as sess:
    #run every operation with variable input
    print ("add result is " + str(sess.run(add, feed_dict={a:2, b:3})))
    print ("multiply result is " + str(sess.run(multiply, feed_dict={a:5, b:4})))

#add result is 5
#multiply result is 20



#Matrix MUltiplication from TensorFlow tutorial
#Create a Constant op that produce a 1*2 matrix. The op is added as a node to the default graph
matrix_a = tf.constant([[3,4]])
#Create another matrix that is 2*1 matrix
matrix_b = tf.constant([[3], [4]])
#create a Matmul op that takes matrix_a, matrix_b as inputs.
#the returned value, "product" represents the result of the matrix multiplication
product = tf.matmul(matrix_a, matrix_b)
#use session run() method to run the matul op
with tf.Session() as sess:
    result = sess.run(product)
    print(result)


#[[25]]





