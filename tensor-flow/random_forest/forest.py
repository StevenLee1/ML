from __future__ import print_function
import tensorflow as tf
from tensorflow.python.ops import resources
from tensorflow.contrib.tensor_forest.python import tensor_forest

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# import data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("dataset", one_hot=False)

# parameters
num_steps = 500 # total steps to train
batch_size = 1024 # the number of samples per batch

num_features = 784 # each image is 28 * 28
num_trees = 10
max_nodes = 1000

# input and target data
X = tf.placeholder(tf.float32, shape=[None, num_features])
# for random forest, Lables must be intergers
Y = tf.placeholder(tf.int32, shape=[None])

# Random Forest Parameters
hparams = tensor_forest.ForestHParams(num_trees=num_trees,
                                      max_nodes=max_nodes).fill()
# build the random forest
forest_graph = tensor_forest.RandomForestGraphs(hparams)
# get training graph and loss
train_op = forest_graph.training_graph(X, Y)
loss_op = forest_graph.training_loss(X, Y)

# measure the accuracy
infer_op, _, = forest_graph.inference_graph(X)
correct_prediction = tf.equal(tf.argmax(infer_op, 1), tf.cast(Y, tf.int64))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# initialize the variables
init_vars = tf.group(tf.global_variables_initializer(),
                     resources.initialize_resources(resources.shared_resources()))

# start tensorflow session
sess = tf.train.MonitoredSession()
# run the initializer
sess.run()

#training
for i in range(1, num_steps + 1):
    # prepare data
    # get the next batch of MNIST data(only images are needed, not labels)
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    _, l = sess.run([train_op, loss_op], feed_dict={X: batch_x, Y: batch_y})
    if i % 50 == 0 or i == 1:
        acc = sess.run(accuracy_op, feed_dict={X: batch_x, Y: batch_y})
        print("Step %i, Loss: %f, Acc: %f" % (i, l, acc))

# test model
test_x, test_y = mnist.test.images, mnist.test.labels
print("Test Accuracy:", sess.run(accuracy_op, feed_dict={X: test_x,
                                                         Y: test_y}))

