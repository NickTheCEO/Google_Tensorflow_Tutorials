# this is a file i did manually

import tensorflow as tf
import numpy as np

# Load and preprocess data
fashion_mnist = tf.keras.datasets.fashion_mnist  # Still using this to load data
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# print(train_images[0].shape) # (28,28) pixel
# print(train_images.shape) # (28,28) pixel, values ranging 0-255
# print(train_labels[0]) # (60000,28,28)
train_images = train_images / 255.0
test_images = test_images / 255.0
#reshape images to vectors
train_images = train_images.reshape(train_images.shape[0], -1)
test_images = test_images.reshape(test_images.shape[0], -1)

#convert labels to one hot encoding
all_labels = np.concatenate(train_labels, test_labels)
numClasses = len(np.unique(all_labels))
train_labels_oneHot = tf.one_hot(train_labels, depth = numClasses)
test_labels_oneHot = tf.one_hot(test_labels, depth = numClasses)

#define model parameters
alpha = 0.01
batch_size = 512
epochs = 2
numNeuronsLayer1 = 128

# define weights and biases, shape and initial values
weights1 = tf.Variable(tf.random.normal([28*28, numNeuronsLayer1], stddev=0.1))
biases1 = tf.Variable(tf.zeros[128])
weights2 = tf.Variable(tf.random.normal([numNeuronsLayer1, numClasses], stddev=0.1))
biases2 = tf.Variable(tf.zeros([numClasses]))

#Define model
def model(X):
    #First layer
    layer1 = X @ weights1 + biases1
    layer1 = tf.nn.relu(layer1)

    #second layer
    logits = layer1 @ weights2 + biases2
    return logits

#Define loss function
def loss_fn(logits, labels):
    return tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels = labels, logits = logits)
    )

#Create optimizer
optimizer = tf.optimizers.Adam(alpha)



