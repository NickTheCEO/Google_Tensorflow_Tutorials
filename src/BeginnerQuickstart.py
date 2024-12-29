"""
This file classifies the MNIST 0-9 digits
"""

# noinspection PyUnresolvedReferences
import tensorflow as tf
import os
print("Tensor flow version:", tf.__version__)

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#scale pixel values of of 0-255 to 0-1, and convert to float
x_train, x_test = x_train / 255.0, x_test / 255.0

def create_model():
    # Define the model architecture
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28,28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model


# predictions = model(x_train[:1]).numpy()
# print(predictions)
# print(tf.nn.softmax(predictions).numpy()) # this normalizes the scores using softmax
# softmax is e^x_i / sum(e^x_i)

#define a loss function
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)

# This untrained model gives probabilities close to random (1/10 for each class), so the initial loss should be close to -tf.math.log(1/10) ~= 2.3.
# print(loss_fn(y_train[:1], predictions).numpy())


# Set up model path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
saved_models_dir = os.path.join(project_root, 'saved_models')
model_path = os.path.join(saved_models_dir, 'quickstart_model.keras')

# Create directory with full permissions if it doesn't exist
if not os.path.exists(saved_models_dir):
    os.makedirs(saved_models_dir, exist_ok=True)

try:
    if os.path.exists(model_path):
        print(f"Found existing model at {model_path}")
        model = tf.keras.models.load_model(model_path)
        print("Model successfully loaded from disk")

        # Evaluate model before additional training
        print("\nEvaluating loaded model:")
        model.evaluate(x_test, y_test, verbose=2)

        # Fine-tune with 1 epoch
        print("\nFine-tuning model with 1 epoch:")
        model.fit(x_train, y_train, epochs=1)
    else:
        print(f"No existing model found at {model_path}")
        print("Training new model...")
        # Create and train new model
        model = create_model()
        model.fit(x_train, y_train, epochs=10)

        # Save the model
        model.save(model_path)
        print(f"Model saved to: {model_path}")

except Exception as e:
    print(f"An error occurred: {str(e)}")
    print("Training model without saving...")
    model = create_model()
    model.fit(x_train, y_train, epochs=5)

# Evaluate the model
print("\nEvaluating model...")
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
print(f"Test accuracy: {test_accuracy}")

#Print probabilities for first 5 entries
probabilities = tf.nn.softmax(model(x_test[:5]).numpy())
rounded_probs = tf.round(probabilities * 100)/100
for i, prob in enumerate(rounded_probs):
    max_index = tf.argmax(prob).numpy()
    max_value = prob[max_index].numpy()
    true_digit = y_test[i]

    print(f"For sample {i+1}, probability = {max_value}, predicted digit = {max_index}, true digit = {true_digit}")
