from __future__ import absolute_import, division, print_function
import numpy as np
import tensorflow as tf

#set Eager API
print('Setting Eager mode')
tf.enable_eager_execution()
tf_eager = tf.contrib.eager

#define constant tensors
print('DEfine constant tensors')
a = tf.constant(2)
print('a=' + str(a))
b = tf.constant(3)
print('b=' + str(b))

#DEfine constant tensors
#a=tf.Tensor(2, shape=(), dtype=int32)
#b=tf.Tensor(3, shape=(), dtype=int32)



#run the operation without the need for tf.Session
c = a + b
print('a+b' + str(c))
d = a * b
print('a*b' + str(d))
#a+btf.Tensor(5, shape=(), dtype=int32)
#a*btf.Tensor(6, shape=(), dtype=int32)




#full compatibility with Numpy
print('Mixing operatons with Tensors and Numpy Arrays')
a = tf.constant([[2,1],
                 [1,0]], dtype=tf.float32)
print("Tensor:")
print(a)
b = np.array([[3,0],
              [5,1]], dtype=np.float32)
print(b)



#run the operation without the need for tf.Session
print('run the operation without the need for tf.Session')
print(a + b)
#tf.Tensor(
#[[5. 1.]
# [6. 1.]], shape=(2, 2), dtype=float32)
print(tf.matmul(a, b))
#tf.Tensor(
#[[11.  1.]
# [ 3.  0.]], shape=(2, 2), dtype=float32)

print('Iterate through Tensor a')
for i in range(a.shape[0]):
    for j in range(a.shape[1]):
        print(a[i][j])

#tf.Tensor(2.0, shape=(), dtype=float32)
#tf.Tensor(1.0, shape=(), dtype=float32)
#tf.Tensor(1.0, shape=(), dtype=float32)
#tf.Tensor(0.0, shape=(), dtype=float32)

