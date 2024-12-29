'''
Rewriting this entire code in TF without taking advantage of Keras
Goal is to learn differences in TF vs Keras
'''

import tensorflow as tf
import numpy as np

# Load and preprocess data
fashion_mnist = tf.keras.datasets.fashion_mnist  # Still using this to load data
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Convert to float32 and normalize
train_images = train_images.astype(np.float32) / 255.0
test_images = test_images.astype(np.float32) / 255.0

# Reshape images to vectors
train_images = train_images.reshape(-1, 28*28)
test_images = test_images.reshape(-1, 28*28)

# Convert labels to one-hot encoding
train_labels = tf.one_hot(train_labels, 10)
test_labels = tf.one_hot(test_labels, 10)

# Define model parameters
learning_rate = 0.001
batch_size = 32
epochs = 2

# Define weights and biases
weights1 = tf.Variable(tf.random.normal([784, 128], stddev=0.1))
biases1 = tf.Variable(tf.zeros([128]))
weights2 = tf.Variable(tf.random.normal([128, 10], stddev=0.1))
biases2 = tf.Variable(tf.zeros([10]))

# Define model function
def model(x):
    # First layer
    layer1 = tf.matmul(x, weights1) + biases1
    layer1 = tf.nn.relu(layer1)

    # Output layer
    logits = tf.matmul(layer1, weights2) + biases2
    return logits

# Define loss function
def loss_fn(logits, labels):
    return tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    )

# Create optimizer
optimizer = tf.optimizers.Adam(learning_rate)

# Training step function
@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x)
        loss = loss_fn(logits, y)

    # Calculate gradients
    gradients = tape.gradient(loss, [weights1, biases1, weights2, biases2])
    # Apply gradients
    optimizer.apply_gradients(zip(gradients, [weights1, biases1, weights2, biases2]))

    return loss

# Training loop
num_batches = len(train_images) // batch_size

for epoch in range(epochs):
    total_loss = 0

    # Shuffle the training data
    indices = tf.random.shuffle(range(len(train_images)))
    train_images_shuffled = tf.gather(train_images, indices)
    train_labels_shuffled = tf.gather(train_labels, indices)

    for batch in range(num_batches):
        start_idx = batch * batch_size
        end_idx = start_idx + batch_size

        x_batch = train_images_shuffled[start_idx:end_idx]
        y_batch = train_labels_shuffled[start_idx:end_idx]

        loss = train_step(x_batch, y_batch)
        total_loss += loss

    avg_loss = total_loss / num_batches
    print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

# Evaluate model
test_logits = model(test_images)
predictions = tf.argmax(test_logits, axis=1)
test_labels_idx = tf.argmax(test_labels, axis=1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, test_labels_idx), tf.float32))
print(f"Test Accuracy: {accuracy:.4f}")
