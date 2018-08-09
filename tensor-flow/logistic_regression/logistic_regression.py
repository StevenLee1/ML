import tensorflow as tf

#import MINST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("dataset/", one_hot=True)
#output:
#Extracting dataset/train-images-idx3-ubyte.gz
#Extracting dataset/train-labels-idx1-ubyte.gz
#Extracting dataset/t10k-images-idx3-ubyte.gz
#Extracting dataset/t10k-labels-idx1-ubyte.gz

