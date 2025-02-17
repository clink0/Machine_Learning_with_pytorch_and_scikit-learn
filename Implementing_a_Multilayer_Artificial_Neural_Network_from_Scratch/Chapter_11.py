"""
Chapter 11: Implementing a Multilayer Artificial Neural Network from Scratch
Luke Bray
February 15, 2025

Explanation Summary:
1. Classifying handwritten digits:
   - Load the MNIST dataset using fetch_openml.
   - Normalize the pixel values and visualize sample images.
   - Split the data into training, validation, and test sets.

2. Implementing a multilayer perceptron:
   - Import a custom neural network class (NeuralNetMLP) for a multilayer perceptron.
   - Define helper functions for activation (sigmoid), one-hot encoding, loss (MSE), and accuracy.
   - Create a minibatch generator for stochastic gradient descent.

3. Neural network training loop:
   - Define functions to compute mean squared error (MSE) and accuracy.
   - Define a training function that loops over epochs and minibatches, performs forward and backward passes,
     and updates weights.
   - Train the neural network while logging loss and accuracy.

4. Evaluating network performance:
   - Plot learning curves (loss and accuracy over epochs).
   - Evaluate the trained model on the test set.
   - Visualize misclassified images from a subset of test data.
"""

################################################################################
# %% Classifying handwritten digits
# This section loads the MNIST dataset, normalizes the pixel values to the range [-1, 1],
# and visualizes a few sample images from different classes. It also splits the dataset
# into training, validation, and test sets.
################################################################################

import pandas as pd  # For data manipulation (not directly used here, but often standard)
from sklearn.datasets import fetch_openml  # To fetch the MNIST dataset

# Fetch the MNIST dataset (70,000 samples, 784 features per image)
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
X = X.values  # Convert to NumPy array
y = y.astype(int).values  # Ensure target labels are integers

print(X.shape)  # Should output (70000, 784)
print(y.shape)  # Should output (70000,)

# Normalize pixel values from [0, 255] to [-1, 1]
X = ((X / 255.) - 0.5) * 2

import matplotlib.pyplot as plt  # For plotting

# Visualize one sample from each digit (0-9)
fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True)
ax = ax.flatten()

for i in range(10):
    # Get the first image for digit i and reshape it to 28x28
    img = X[y == i][0].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys')

# Remove axis ticks for a cleaner display
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()

# Visualize 25 images of digit 7
fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True, figsize=(8, 8))
ax = ax.flatten()

for i in range(25):
    img = X[y == 7][i].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys')

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()

from sklearn.model_selection import train_test_split

# Split the dataset:
# - X_temp and X_test: 10000 samples reserved for test
# - Then, further split X_temp into training (remaining) and validation (5000 samples)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=10000, random_state=123, stratify=y
)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_temp, y_temp, test_size=5000, random_state=123, stratify=y_temp
)

################################################################################
# %% Implementing a multilayer perceptron
# This section imports a custom multilayer neural network class (NeuralNetMLP),
# instantiates the model with 50 hidden units for a 10-class classification problem,
# and prepares utility functions such as sigmoid activation and one-hot encoding.
################################################################################

from neuralnet import NeuralNetMLP  # Custom neural network class

# Initialize the neural network with:
# - Input layer size = 28x28 pixels,
# - 50 hidden units,
# - 10 output classes (digits 0-9)
model = NeuralNetMLP(
    num_features=28*28,
    num_hidden=50,
    num_classes=10
)

# Define the sigmoid activation function
def sigmoid(z):
    return 1. / (1. + np.exp(-z))

# Function to convert integer labels to one-hot encoded vectors
def int_to_onehot(y, num_labels):
    ary = np.zeros((y.shape[0], num_labels))
    for i, val in enumerate(y):
        ary[i, val] = 1
    return ary

import numpy as np

# Set training parameters
num_epochs = 50
minibatch_size = 100

# Define a generator that shuffles and yields minibatches from the data
def minibatch_generator(X, y, minibatch_size):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    for start_idx in range(0, indices.shape[0] - minibatch_size + 1, minibatch_size):
        batch_idx = indices[start_idx:start_idx + minibatch_size]
        yield X[batch_idx], y[batch_idx]

# Test the minibatch generator by getting one minibatch (for debugging)
for i in range(num_epochs):
    minibatch_gen = minibatch_generator(X_train, y_train, minibatch_size)
    for X_train_mini, y_train_mini in minibatch_gen:
        break
    break

print(X_train_mini.shape)  # Expected shape: (minibatch_size, 784)
print(y_train_mini.shape)  # Expected shape: (minibatch_size,)

# Define mean squared error loss for the network
def mse_loss(targets, probas, num_labels=10):
    onehot_targets = int_to_onehot(targets, num_labels=num_labels)
    return np.mean((onehot_targets - probas) ** 2)

# Define accuracy as the proportion of correct predictions
def accuracy(targets, predicted_labels):
    return np.mean(predicted_labels == targets)

# Perform a forward pass on the validation set and compute initial loss and accuracy
_, probas = model.forward(X_valid)
mse = mse_loss(y_valid, probas)
print(f'Initial validation MSE: {mse:.1f}')

# Convert network outputs to predicted labels
predicted_labels = np.argmax(probas, axis=1)
acc = accuracy(y_valid, predicted_labels)
print(f'Initial validation accuracy: {acc*100:.1f}%')

# Function to compute mean squared error and accuracy over the dataset using minibatches
def compute_mse_and_acc(nnet, X, y, num_labels=10, minibatch_size=100):
    mse, correct_pred, num_examples = 0, 0, 0
    minibatch_gen = minibatch_generator(X, y, minibatch_size)
    for i, (features, targets) in enumerate(minibatch_gen):
        _, probas = nnet.forward(features)
        predicted_labels = np.argmax(probas, axis=1)
        onehot_targets = int_to_onehot(targets, num_labels=num_labels)
        loss = np.mean((onehot_targets - probas)**2)
        correct_pred += (predicted_labels == targets).sum()
        num_examples += targets.shape[0]
        mse += loss
    mse = mse / i
    acc = correct_pred / num_examples
    return mse, acc

# Compute initial validation MSE and accuracy using the helper function
mse, acc = compute_mse_and_acc(model, X_valid, y_valid)
print(f'Initial validation MSE: {mse:.1f}%')
print(f'Initial validation accuracy: {acc*100:.1f}%')

################################################################################
# %% Coding the neural network training loop
# This section defines a training function for the neural network. For each epoch,
# it iterates over minibatches, performs forward and backward propagation,
# and updates the network parameters using gradient descent. It also computes and
# logs the training and validation loss and accuracy for each epoch.
################################################################################

def train(model, X_train, y_train, X_valid, y_valid, num_epochs, learning_rate=0.1):
    epoch_loss = []         # To store loss for each epoch
    epoch_train_acc = []    # To store training accuracy for each epoch
    epoch_valid_acc = []    # To store validation accuracy for each epoch

    for e in range(num_epochs):
        # Generate minibatches from training data
        minibatch_gen = minibatch_generator(X_train, y_train, minibatch_size)
        # Loop over each minibatch and perform one training step
        for X_train_mini, y_train_mini in minibatch_gen:
            # Forward pass through the network: get hidden and output activations
            a_h, a_out = model.forward(X_train_mini)
            # Backward pass: compute gradients with respect to all weights and biases
            d_loss__d_w_out, d_loss__d_b_out, d_loss__d_w_h, d_loss__d_b_h = model.backward(X_train_mini, a_h, a_out, y_train_mini)
            # Update weights and biases with gradient descent step
            model.weight_h -= learning_rate * d_loss__d_w_h
            model.bias_h   -= learning_rate * d_loss__d_b_h
            model.weight_out -= learning_rate * d_loss__d_w_out
            model.bias_out   -= learning_rate * d_loss__d_b_out

        # After each epoch, compute loss and accuracy on both training and validation sets
        train_mse, train_acc = compute_mse_and_acc(model, X_train, y_train)
        valid_mse, valid_acc = compute_mse_and_acc(model, X_valid, y_valid)

        # Convert accuracies to percentages
        train_acc, valid_acc = train_acc * 100, valid_acc * 100
        epoch_train_acc.append(train_acc)
        epoch_valid_acc.append(valid_acc)
        epoch_loss.append(train_mse)

        print(f'Epoch: {e+1:03d}/{num_epochs:03d}'
              f'| Train MSE: {train_mse:.2f} '
              f'| Train Acc: {train_acc:.2f}% '
              f'| Valid Acc: {valid_acc:.2f}%')

    return epoch_loss, epoch_train_acc, epoch_valid_acc

# Set random seed for reproducibility
np.random.seed(123)
# Train the neural network model using the training and validation sets
epoch_loss, epoch_train_acc, epoch_valid_acc = train(
    model,
    X_train,
    y_train,
    X_valid,
    y_valid,
    num_epochs=50,
    learning_rate=0.1
)

################################################################################
# %% Evaluating the neural network performance
# This section visualizes the learning curves (loss and accuracy) over epochs,
# evaluates the model on the test set, and displays some misclassified test images.
################################################################################

# Plot the training loss curve over epochs
plt.plot(range(len(epoch_loss)), epoch_loss)
plt.ylabel('Mean Squared Error')
plt.xlabel('Epoch')
plt.show()

# Plot training and validation accuracy over epochs
plt.plot(range(len(epoch_train_acc)), epoch_train_acc, label='Training')
plt.plot(range(len(epoch_valid_acc)), epoch_valid_acc, label='Validation')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='lower right')
plt.show()

# Evaluate the trained model on the test set and print test accuracy
test_mse, test_acc = compute_mse_and_acc(model, X_test, y_test)
print(f'Test accuracy: {test_acc*100:.2f}%')

# Select a subset of test data for detailed analysis (first 1000 samples)
X_test_subset = X_test[:1000, :]
y_test_subset = y_test[:1000]

# Compute the network output (probas) on the test subset
_, probas = model.forward(X_test_subset)
# Convert probabilities to predicted labels
test_pred = np.argmax(probas, axis=1)

# Identify misclassified images and corresponding labels (take first 25)
misclassified_images = X_test_subset[y_test_subset != test_pred][:25]
misclassified_labels = test_pred[y_test_subset != test_pred][:25]
correct_labels = y_test_subset[y_test_subset != test_pred][:25]

# Plot the misclassified images in a 5x5 grid with true and predicted labels
fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True, figsize=(8, 8))
ax = ax.flatten()

for i in range(25):
    img = misclassified_images[i].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')
    ax[i].set_title(f'{i+1}\nTrue: {correct_labels[i]}\nPredicted: {misclassified_labels[i]}')

# Remove axis ticks for a cleaner look
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()
