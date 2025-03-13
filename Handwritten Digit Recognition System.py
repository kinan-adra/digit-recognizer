# Import necessary libraries
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load and explore the MNIST dataset
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Step 2: Preprocess the data
train_images = train_images / 255.0  # Normalize pixel values to 0-1
test_images = test_images / 255.0
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))  # Add channel dimension
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

# Step 3: Build the neural network model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),  # Convolutional layer
    layers.MaxPooling2D((2, 2)),  # Pooling layer
    layers.Flatten(),  # Flatten the 2D image data to 1D
    layers.Dense(128, activation='relu'),  # Fully connected layer
    layers.Dense(10, activation='softmax')  # Output layer (10 digits)
])

# Step 4: Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Step 5: Train the model
model.fit(train_images, train_labels, epochs=5, batch_size=32)

# Step 6: Evaluate the model
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_accuracy:.4f}")

# Step 7: Make predictions and visualize
predictions = model.predict(test_images)
for i in range(5):  # Show first 5 test images and predictions
    plt.imshow(test_images[i].reshape(28, 28), cmap='gray')
    plt.title(f"Predicted: {np.argmax(predictions[i])}, Actual: {test_labels[i]}")
    plt.savefig(f'example{i}.png')
    plt.show()

# Step 8: Save the model
model.save("mnist_digit_recognizer.h5")
print("Model saved as 'mnist_digit_recognizer.h5'")