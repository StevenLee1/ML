from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow.contrib.factorization import KMeans

# ignore all GPUs, tf random forest does not benefit from it
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ""

#import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('dataset', one_hot=True)
full_data_x = mnist.train.images

# parameters
num_steps = 50 # total steps to train
batch_size = 1024 # the number of samples per batch
k = 25 # the number of clusters
num_classes = 10 # the 10 digits
num_features = 784 # each image is 28 * 28

# input images
X = tf.placeholder(tf.float32, shape=[None, num_features])
# Labels (for assigning a label to a centroid and testing)
Y = tf.placeholder(tf.float32, shape=[None, num_classes])

# K-means parameters
kmeans = KMeans(inputs=X, num_clusters=k,
                distance_metric='cosine',
                use_mini_batch=True)

# Build KMeans graph
(all_scores, cluster_idx, scores, cluster_centers_initialized,
 cluster_centers_vars, init_op, train_op) = kmeans.training_graph()
avg_distance = tf.reduce_mean(scores)

# initialize the variables
init_vars = tf.global_variables_initializer()

# start tensorflow session
sess = tf.Session()
# run the initializer
sess.run(init_vars, feed_dict={X: full_data_x})
sess.run(init_op, feed_dict={X: full_data_x})

#training
for i in range(1, num_steps + 1):
    _, d, idx = sess.run([train_op, avg_distance, cluster_idx],
                         feed_dict={X: full_data_x})
    if i % 10 == 0 or i == 1:
        print("Step %i, Avg Distance: %f" % (i,d))
