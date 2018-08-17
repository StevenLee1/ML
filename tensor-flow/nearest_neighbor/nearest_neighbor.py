import numpy as np
import tensorflow as tf

#import MINST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('dataset', one_hot=True)

#we limit mnist data in this example
Xtr, Ytr = mnist.train.next_batch(5000) #5000 for training
Xte, Yte = mnist.test.next_batch(200) #200 for testing

#tf Graph Input
xtr = tf.placeholder('float', [None, 784])
xte = tf.placeholder('float', [784])

#Nearest Neighbor calculation using L1 distance
#Calculate L1 distance
distance = tf.reduce_sum(tf.abs(tf.add(xtr, tf.negative(xte))),
                         reduction_indices=1)
#predict: Get min distance index(Nearest neighbor)
pred = tf.argmin(distance, 0)
accuracy = 0

#INitialize the variables(assign default values)
init = tf.global_variables_initializer()
#Start training
with tf.Session() as sess:
    sess.run(init)

    #loop over test data
    for i in range(len(Xte)):
        # get nearest neighbor
        nn_index = sess.run(pred, feed_dict={xtr:Xtr,
                                             xte:Xte[i,:]})
        #get nearest neighbor class label and compare it to its true label
        print("Test", i, "Prediction:", np.argmax(Ytr[nn_index]),
              "True Class", np.argmax(Yte[i]))
        #calculate accuracy
        if np.argmax(Ytr[nn_index] == np.argmax(Yte[i])):
            accuracy += 1.0/len(Xte)
    print("Done")
    print("Accuracy:", accuracy)
