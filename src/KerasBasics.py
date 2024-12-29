import tensorflow as tf
import numpy as np

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# print(train_images.shape) #shape = (60000, 28, 28), 60k images 28x28p
# print(train_labels) #array([9, 0, 0, ..., 3, 0, 5], dtype=uint8)
# print(test_images.shape) #(10000, 28, 28) 10k images 28x28p

#View first image
# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()

#Preprocess the data
#Scale -0-255 pixel values to 0-1
train_images = train_images / 255.0
test_images = test_images / 255.0

#View first 25 images
# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()

#build and compile model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape = (28,28)),
    tf.keras.layers.Dense(128, activation ='relu'),
    tf.keras.layers.Dense(10)
])
model.compile(optimizer = 'adam',
              loss= tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
              metrics = ['accuracy'])
model.fit(train_images, train_labels, epochs = 2)

#evaluate accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose = 2)
print(f"Test Accuracy: {test_acc}")
#Note test accuracy > train accuracy, resulting in overfitting

# Do probability predictions
probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()]) #append softmax layer to already trained model, to convert to probabilities
predictions = probability_model.predict(test_images)

#use the trained model
img = test_images[1]
# print(img.shape) (28,28)

img = (np.expand_dims(img,0)) #Turning single image into batch size of 1, by adding third dimension at beginning, hence the 0. (28,28)->(1,28,28)
# print(img.shape) (1, 28,28)

predictions_single = probability_model.predict(img)
print(predictions_single)
