################################################################################
# Title: Machine Learning with PyTorch and Scikit-Learn: Chapter 3
# Author: Luke Bray
# Date: January 30, 2025
#
# Explanation Summary:
#
# 1. Create Datasets
#    - Loads the iris dataset and extracts features/targets.
#
# 2. Split Test and Train
#    - Divides the data into training and test sets, preserving class ratios.
#
# 3. Standardize Data
#    - Normalizes features to have zero mean and unit variance.
#
# 4. Train Perceptron
#    - Trains a Perceptron classifier on standardized data.
#
# 5. Predict
#    - Uses the trained Perceptron to predict test labels and count errors.
#
# 6. Accuracy
#    - Compares predicted labels to actual labels and computes accuracy.
#
# 7. Plot of Sigmoid Function
#    - Demonstrates the shape of the logistic (sigmoid) function.
#
# 8. Plot of Logistic Loss
#    - Visualizes negative log-likelihood loss functions (for y=0 and y=1).
#
# 9. Plot Decision Regions
#    - Helper function to visualize 2D decision boundaries from any classifier.
#
# 10. Logistic Regression Class (Custom GD)
#     - Implements logistic regression using manual gradient descent updates.
#
# 11. Scikit-Learn Optimized Logistic Regression
#     - Demonstrates multi-class logistic regression via scikit-learn.
#
# 12. Plot Regularization vs C
#     - Shows how varying the regularization strength (C) changes model weights.
#
# 13. Support Vector Machine
#     - Trains an SVM with a linear kernel and plots decision boundaries.
#
# 14. SGD Alternatives
#     - Illustrates how SGDClassifier can emulate perceptron, logistic, or hinge loss.
#
# 15. XOR Dataset
#     - Creates a synthetic "XOR" dataset for non-linear classification demos.
#
# 16. Entropy
#     - Plots the binary entropy for different class probabilities.
#
# 17. Impurity Indices
#     - Compares Gini impurity, entropy, and misclassification error as p changes.
#
# 18. Decision Tree
#     - Trains a decision tree (Gini criterion) on the Iris dataset and visualizes it.
#
# 19. Random Forest
#     - Uses an ensemble of decision trees (random forest) for classification.
#
# 20. K-Nearest Neighbors
#     - Implements KNN with Euclidean distance on standardized data and visualizes results.
################################################################################


###############################################################################
# %% CREATE DATASETS
# This section loads the Iris dataset from scikit-learn, extracts two specific
# features (petal length and petal width), and obtains the target class labels.
# It also prints out the unique class labels to ensure everything is loaded
# correctly.
###############################################################################

from sklearn import datasets  # Importing the datasets module from scikit-learn
from sklearn.model_selection import train_test_split  # For splitting data
from sklearn.preprocessing import StandardScaler  # For data standardization
from sklearn.linear_model import Perceptron  # The Perceptron classifier
from sklearn.metrics import accuracy_score  # For measuring prediction accuracy
from sklearn.linear_model import LogisticRegression  # The Logistic Regression model
import matplotlib.pyplot as plt  # For plotting
import numpy as np  # For numerical computations

# Load Iris dataset
iris = datasets.load_iris()  # Loads the Iris dataset from scikit-learn

# Extract features (Petal length, Petal width)
X = iris.data[:, [2, 3]]  # Slices the data to get columns 2 and 3 (petal length, petal width)

# Get target classes
y = iris.target  # The target labels (class labels)

# Print unique class labels
print('Class labels: ', np.unique(y))  # np.unique(y) shows the distinct classes in y

###############################################################################
# %% SPLIT TEST AND TRAIN DATA
# Here, we split the dataset into training and test sets, ensuring the class
# distributions remain similar (via stratification). Then, we verify that the
# training and test sets have the expected distribution of labels.
###############################################################################

# Split the data into training and testing sets.
# test_size=0.3 means 30% of data is used as test set.
# random_state=1 ensures reproducibility.
# stratify=y preserves class distribution across sets.
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=1,
                                                    stratify=y)

# Print label counts in the full dataset
print('Labels counts in y: ', np.bincount(y))
# np.bincount(y) counts the number of occurrences of each class label

# Print label counts in the training set
print('Labels count in y_train: ', np.bincount(y_train))

# Print label counts in the test set
print('Labels counts in y_test: ', np.bincount(y_test))

###############################################################################
# %% STANDARDIZE DATA
# Standardizing the features so they have zero mean and unit variance. This is
# often done before training many machine learning models to improve performance.
###############################################################################

# Create a StandardScaler object
sc = StandardScaler()

# Compute the mean and std for scaling based on the training data
sc.fit(X_train)

# Transform (standardize) the training data
X_train_std = sc.transform(X_train)

# Transform (standardize) the test data using the same parameters
X_test_std = sc.transform(X_test)

###############################################################################
# %% TRAIN PERCEPTRON
# We instantiate a Perceptron classifier, train it on the standardized training
# data, and then use it to predict labels for the test set.
###############################################################################

# Create a Perceptron object with a learning rate (eta0) and fixed random_state
ppn = Perceptron(eta0=0.1, random_state=1)

# Train (fit) the Perceptron on the standardized training data
ppn.fit(X_train_std, y_train)

###############################################################################
# %% PREDICT
# Here we use the trained Perceptron to predict the test set labels, then we
# count how many are misclassified.
###############################################################################

# Generate predictions for the test data
y_pred = ppn.predict(X_test_std)

# Print the number of misclassified examples
print('Misclassified examples: %d' % (y_test != y_pred).sum())

###############################################################################
# %% ACCURACY
# We measure how accurate the model is by comparing predictions to the actual
# labels. We do this in two different ways: using accuracy_score and the model's
# own .score() method, which returns the same result.
###############################################################################

# Compute and print the accuracy using accuracy_score
print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))

# Print the accuracy using the Perceptron object's score() method
print('Accuracy: %.3f' % ppn.score(X_test_std, y_test))

###############################################################################
# %% GET MORE INFO ON PERCEPTRON CLASS
# Here we just call Python's help() function on the Perceptron class to get
# documentation in an interactive environment (will print docstring).
###############################################################################

help(Perceptron)


###############################################################################
# %% PLOT OF SIGMOID FUNCTION
# This section defines a sigmoid function for any real value z, then creates a
# range of z values and plots the sigmoid output to illustrate how the function
# behaves. The vertical line at x=0 and the y-axis grid are just for reference.
###############################################################################

def sigmoid(z):
    # Computes the logistic sigmoid function 1 / (1 + e^-z)
    return 1.0 / (1.0 + np.exp(-z))


# Create a range of values from -7 to 7 (step=0.1)
z = np.arange(-7, 7, 0.1)

# Calculate the sigmoid for each value in z
sigma_z = sigmoid(z)

# Plot the sigmoid curve
plt.plot(z, sigma_z)

# Draw a vertical line at z=0.0
plt.axvline(0.0, color='k')

# Set the y-axis limits
plt.ylim(-0.1, 1.1)

# Label the axes
plt.xlabel('z')
plt.ylabel('$\sigma (z)$')

# Set the y-axis tick marks
plt.yticks([0.0, 0.5, 1.0])

# Get current axes
ax = plt.gca()

# Draw horizontal grid lines
ax.yaxis.grid(True)

# Adjust subplot parameters to give some padding
plt.tight_layout()

# Display the figure
plt.show()


###############################################################################
# %% PLOT OF LOSS FUNCTION USED IN LOGISTIC REGRESSION
# We define two functions that compute the negative log-likelihood loss for
# y=1 (loss_1) and y=0 (loss_0). We then plot how these loss values change
# depending on the output of the sigmoid function.
###############################################################################

def loss_1(z):
    # Negative log-likelihood if the true label y=1
    return - np.log(sigmoid(z))


def loss_0(z):
    # Negative log-likelihood if the true label y=0
    return - np.log(1 - sigmoid(z))


# Create a range of z values from -10 to 10
z = np.arange(-10, 10, 0.1)

# Apply the sigmoid function
sigma_z = sigmoid(z)

# Compute loss for y=1
c1 = [loss_1(x) for x in z]

# Plot the loss for y=1
plt.plot(sigma_z, c1, label='L(w, b) if y=1')

# Compute loss for y=0
c0 = [loss_0(x) for x in z]

# Plot the loss for y=0
plt.plot(sigma_z, c0, linestyle='--', label='L(w, b) if y=0')

# Set y-limit
plt.ylim(0.0, 5.1)

# Set x-limit
plt.xlim([0, 1])

# Label the axes
plt.xlabel('$\sigma(z)$')
plt.ylabel('L(w, b)')

# Add a legend
plt.legend(loc='best')

# Adjust layout
plt.tight_layout()

# Show the figure
plt.show()

###############################################################################
# %% PLOT DECISION REGIONS
# This function plots decision boundaries for a classifier on a 2D dataset.
# It uses a meshgrid to create a region of points, predicts each point's label,
# and then uses a contour plot to show the boundaries between classes. It also
# highlights which points were used as the test set (if provided).
###############################################################################

from matplotlib.colors import ListedColormap  # For custom color maps
import matplotlib.pyplot as plt  # For plotting


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    # Markers and colors for plotting
    markers = ('o', 's', '^', 'v', '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # Determine the min and max values for the two features
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # Create a meshgrid across these min and max values
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))

    # Predict class labels for all points on the meshgrid
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)

    # Reshape the predictions to match the shape of the meshgrid
    lab = lab.reshape(xx1.shape)

    # Plot the contour filled map using the predictions
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)

    # Set the x and y limits
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # Plot the data points for each class
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=f'Class {cl}',
                    edgecolor='black')

    # If test_idx is provided, highlight the test set examples
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1],
                    c='none', edgecolor='black', alpha=1.0,
                    linewidth=1, marker='o',
                    s=100, label='Test set')


###############################################################################
# %% LOGISTIC REGRESSION CLASS (GD IMPLEMENTATION)
# We create a custom Logistic Regression class that uses gradient descent to
# update the weights and bias. We define fit(), predict(), activation(), etc.
# Then we train this custom model on a subset of the training data (classes 0
# and 1) and visualize its decision regions.
###############################################################################

class LogisticRegressionGD:
    """
    Gradient descent-based logistic regression classifier.

    Parameters
    ----------
    eta (float): Learning rate (between 0.0 and 1.0)
    n_iter (int): Passes over the training dataset
    random_state (int): Random number generator seed for random weight initialization

    Attributes
    ----------
    w_ (1d-array): Weights after training
    b_ (scalar): Bias unit after fitting
    losses_ (list): Mean squared error loss function values in each epoch
    """

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta  # Learning rate
        self.n_iter = n_iter  # Number of iterations
        self.random_state = random_state  # Seed for random weight init

    def fit(self, X, y):
        """
        Fit training data
        """
        # Set random seed
        rgen = np.random.RandomState(self.random_state)

        # Initialize weights (small random numbers)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])

        # Initialize bias to 0
        self.b_ = np.float64(0.)

        # Keep track of loss values after each epoch
        self.losses_ = []

        # Gradient descent loop
        for i in range(self.n_iter):
            # Calculate net input
            net_input = self.net_input(X)

            # Apply sigmoid activation
            output = self.activation(net_input)

            # Calculate error (y - output)
            errors = (y - output)

            # Update weights (using derivative of loss w.r.t. w)
            self.w_ += self.eta * 2.0 * X.T.dot(errors) / X.shape[0]

            # Update bias (using the mean error)
            self.b_ += self.eta * 2.0 * errors.mean()

            # Compute the logistic loss for this epoch and store
            loss = (-y.dot(np.log(output)) - ((1 - y).dot(np.log(1 - output))) / X.shape[0])
            self.losses_.append(loss)

        return self

    def net_input(self, X):
        """
        Calculate net input = X * w + b
        """
        return np.dot(X, self.w_) + self.b_

    def activation(self, z):
        """
        Apply the sigmoid function in a numerically stable way (clipping).
        """
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def predict(self, X):
        """
        Return class label after unit step: 1 if sigmoid >= 0.5, else 0
        """
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)


# Select only class 0 and class 1 (binary classification subset)
X_train_01_subset = X_train_std[(y_train == 0) | (y_train == 1)]
y_train_01_subset = y_train[(y_train == 0) | (y_train == 1)]

# Instantiate and train the custom Logistic Regression
lrgd = LogisticRegressionGD(eta=0.3,
                            n_iter=1000,
                            random_state=1)
lrgd.fit(X_train_01_subset, y_train_01_subset)

# Plot decision regions for this custom model
plot_decision_regions(X=X_train_01_subset,
                      y=y_train_01_subset,
                      classifier=lrgd)
plt.xlabel('Petal length [standardized]')
plt.ylabel('Petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

###############################################################################
# %% SCIKIT-LEARN OPTIMIZED MULTICLASS LOGISTIC REGRESSION MODEL
# Here we use scikit-learn's LogisticRegression with 'ovr' (one-vs-rest) mode
# to handle multiclass problems. We plot the decision regions on the combined
# train/test set.
###############################################################################

# Combine train and test data for plotting
X_combined_std = np.vstack((X_train_std, X_test_std))  # Stack arrays in vertical direction
y_combined = np.hstack((y_train, y_test))  # Stack arrays in horizontal direction

# Create logistic regression model with higher C => weaker regularization
lr = LogisticRegression(C=100.0, solver='lbfgs', multi_class='ovr')

# Fit the model on the training set
lr.fit(X_train_std, y_train)

# Plot decision regions on the combined dataset, highlighting test samples
plot_decision_regions(X_combined_std,
                      y_combined,
                      classifier=lr,
                      test_idx=range(105, 150))
plt.xlabel('Petal length [standardized]')
plt.ylabel('Petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

# Predicting class labels for a small subset of the test data
indirect_class_label_prediction = lr.predict_proba(X_test_std[:3, :]).argmax(axis=1)
# .predict_proba(...) returns the class probabilities, .argmax(axis=1) gets the class with the highest probability

more_direct_class_label_prediction = lr.predict(X_test_std[:3, :])
# .predict(...) directly returns the predicted class labels


###############################################################################
# %% PLOTTING REGULARIZATION AS A FUNCTION OF C
# We train a logistic regression model repeatedly with different values of C,
# and record the resulting weight coefficients. Then we plot these coefficients
# as a function of C to see how strong vs weak regularization affects them.
###############################################################################

weights, params = [], []
for c in np.arange(-5, 5):
    # For each exponent, we compute 10**c
    lr = LogisticRegression(C=10. ** c, multi_class='ovr')
    lr.fit(X_train_std, y_train)
    # Store the coefficients of class "1" for each model
    weights.append(lr.coef_[1])
    params.append(10. ** c)

# Convert list of weights to NumPy array for easier plotting
weights = np.array(weights)

# Plot how the weight coefficients for petal length and petal width change
plt.plot(params, weights[:, 0], label='Petal length')
plt.plot(params, weights[:, 1], linestyle='--', label='Petal width')

# Labeling the axes
plt.ylabel('Weight coefficient')
plt.xlabel('C')

# Adding a legend
plt.legend(loc='upper left')

# Using log-scale for x-axis since C is multiplied by powers of 10
plt.xscale('log')

# Show the plot
plt.show()

###############################################################################
# %% IMPLEMENT SUPPORT VECTOR MACHINE
# We train a Support Vector Machine (SVM) with a linear kernel on the standardized
# iris dataset and then visualize its decision boundary.
###############################################################################

from sklearn.svm import SVC  # Importing Support Vector Classifier

# Create an SVC with linear kernel
svm = SVC(kernel='linear', C=1.0, random_state=1)

# Fit SVM to training data
svm.fit(X_train_std, y_train)

# Plot decision regions using the combined dataset
plot_decision_regions(X_combined_std,
                      y_combined,
                      classifier=svm,
                      test_idx=range(105, 150))
plt.xlabel('Petal length [standardized]')
plt.ylabel('Petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

###############################################################################
# %% ALTERNATIVE IMPLEMENTATIONS (SGD CLASSIFIERS)
# Demonstration of scikit-learn's SGDClassifier with different loss functions:
# 'perceptron' (Perceptron), 'log' (Logistic Regression), 'hinge' (SVM).
###############################################################################

from sklearn.linear_model import SGDClassifier  # Stochastic Gradient Descent

# SGD Classifier with perceptron loss
ppn = SGDClassifier(loss='perceptron')

# SGD Classifier with logistic loss
lr = SGDClassifier(loss='log')

# SGD Classifier with hinge loss (SVM)
svm = SGDClassifier(loss='hinge')

###############################################################################
# %% CREATE XOR DATASET WITH RANDOM NOISE
# Generating a synthetic "XOR" dataset with random noise, then plotting it to
# visualize class 0 and class 1 in a 2D feature space.
###############################################################################

import matplotlib.pyplot as plt
import numpy as np

# Set random seed for reproducibility
np.random.seed(1)

# Generate 200 samples of 2D features
X_xor = np.random.randn(200, 2)

# XOR condition for labeling: whether x>0 is different than y>0
y_xor = np.logical_xor(X_xor[:, 0] > 0,
                       X_xor[:, 1] > 0)

# Convert boolean to integer labels (1/0)
y_xor = np.where(y_xor, 1, 0)

# Scatter plot of the XOR dataset
plt.scatter(X_xor[y_xor == 1, 0],
            X_xor[y_xor == 1, 1],
            c='royalblue', marker='s',
            label='Class 1')

plt.scatter(X_xor[y_xor == 0, 0],
            X_xor[y_xor == 0, 1],
            c='tomato', marker='o',
            label='Class 0')

# Set axis limits
plt.xlim([-3, 3])
plt.ylim([-3, 3])

# Label the axes
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Show legend
plt.legend(loc='best')

# Adjust layout
plt.tight_layout()

# Display the figure
plt.show()

###############################################################################
# %% SVM (RBF KERNEL) FOR XOR DATA
# We train a Support Vector Machine with an RBF (Gaussian) kernel on the XOR
# dataset. Because XOR is not linearly separable, an RBF kernel helps create
# a non-linear decision boundary.
###############################################################################

# Create an SVC with RBF kernel, gamma=0.10, and C=10.0
svm = SVC(kernel='rbf', random_state=1, gamma=0.10, C=10.0)

# Fit to the XOR data
svm.fit(X_xor, y_xor)

# Plot decision regions on the XOR data
plot_decision_regions(X_xor, y_xor, classifier=svm)
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

###############################################################################
# %% SVM (RBF KERNEL) ON IRIS DATA
# We train two different RBF SVMs on the iris dataset but vary the gamma value
# to see how this impacts the decision boundary (e.g. more complex boundaries
# with a larger gamma).
###############################################################################

# SVC with RBF kernel, gamma=0.2
svm = SVC(kernel='rbf', random_state=1, gamma=0.2, C=1.0)
svm.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std,
                      y_combined,
                      classifier=svm,
                      test_idx=range(105, 150))
plt.xlabel('Petal length [standardized]')
plt.ylabel('Petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

# SVC with RBF kernel, gamma=100.0
svm = SVC(kernel='rbf', random_state=1, gamma=100.0, C=1.0)
svm.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std,
                      y_combined,
                      classifier=svm,
                      test_idx=range(105, 150))
plt.xlabel('Petal length [standardized]')
plt.ylabel('Petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()


###############################################################################
# %% ENTROPY VALUES FOR DIFFERENT CLASS-MEMBERSHIP PROBABILITIES
# This code defines an entropy function for a binary classification setting and
# plots how the entropy changes with different values of p. Higher entropy means
# more uncertainty in the classification.
###############################################################################

def entropy(p):
    # Calculate binary entropy for probability p
    return - p * np.log2(p) - (1 - p) * np.log2((1 - p))


# Create a range of p-values from 0 to 1 (step=0.01)
x = np.arange(0.0, 1.0, 0.01)

# Calculate entropy for each p in x, skipping any p=0 to avoid log(0)
ent = [entropy(p) if p != 0 else None for p in x]

# Label the axes
plt.ylabel('Entropy')
plt.xlabel('Class-membership probability p(i=1)')

# Plot the entropy curve
plt.plot(x, ent)

# Show the plot
plt.show()

###############################################################################
# %% DIFFERENT IMPURITY INDICES FOR CLASS-MEMBERSHIP PROBABILITIES
# This compares three common impurity measures (Gini, Entropy, Misclassification
# error) for values of p between 0 and 1. The plot helps illustrate how each
# behaves as the probability shifts.
###############################################################################

import matplotlib.pyplot as plt
import numpy as np


def gini(p):
    # Binary Gini impurity
    return p * (1 - p) + (1 - p) * (1 - (1 - p))


def entropy(p):
    # Binary entropy
    return - p * np.log2(p) - (1 - p) * np.log2((1 - p))


def error(p):
    # Misclassification error
    return 1 - np.max([p, 1 - p])


x = np.arange(0.0, 1.0, 0.01)

# Compute entropy for each p, avoid log(0)
ent = [entropy(p) if p != 0 else None for p in x]

# Scale entropy by 0.5 for demonstration
sc_ent = [e * 0.5 if e else None for e in ent]

# Compute misclassification error
err = [error(i) for i in x]

# Start plotting
fig = plt.figure()
ax = plt.subplot(111)

# Plot all impurity measures
for i, lab, ls, c in zip([ent, sc_ent, gini(x), err],
                         ['Entropy', 'Entropy (scaled)',
                          'Gini impurity',
                          'Misclassification error'],
                         ['-', '-', '--', '-.'],
                         ['black', 'lightgray', 'red', 'green']):
    line = ax.plot(x, i, label=lab,
                   linestyle=ls, lw=2, color=c)

# Place a legend above the plot
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
          ncol=5, fancybox=True, shadow=False)

# Draw horizontal dashed lines at 0.5 and 1.0
ax.axhline(y=0.5, linewidth=1, color='k', linestyle='--')
ax.axhline(y=1.0, linewidth=1, color='k', linestyle='--')

# Set y-limit
plt.ylim([0, 1.1])

# Label the axes
plt.xlabel('p(i=1)')
plt.ylabel('impurity index')

# Show the figure
plt.show()

###############################################################################
# %% DECISION TREE IMPLEMENTATION
# This section demonstrates how to train a decision tree with a Gini impurity
# criterion on the (unstandardized) iris data. We then plot the resulting
# decision regions and examine the decision tree structure.
###############################################################################

from sklearn.tree import DecisionTreeClassifier  # Import for decision tree

# Instantiate a decision tree model
tree_model = DecisionTreeClassifier(criterion='gini',
                                    max_depth=4,
                                    random_state=1)

# Train the model on the training data (unstandardized here)
tree_model.fit(X_train, y_train)

# Combine the full dataset
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

# Plot decision regions for the decision tree
plot_decision_regions(X_combined,
                      y_combined,
                      classifier=tree_model,
                      test_idx=range(105, 150))
plt.xlabel('Petal length [cm]')
plt.ylabel('Petal width [cm]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

###############################################################################
# %% DECISION TREE VISUALIZATION
# We use sklearn's tree.plot_tree to visualize the trained decision tree.
# The 'filled=True' option adds color shading that indicates majority classes.
###############################################################################

from sklearn import tree  # For tree visualization

# Custom feature names for clarity
feature_names = ['Sepal length', 'Sepal Width',
                 'Petal length', 'Petal width']

# Plot the tree
tree.plot_tree(tree_model, feature_names=feature_names, filled=True)
plt.show()

###############################################################################
# %% RANDOM FOREST IMPLEMENTATION
# We train a random forest (ensemble of decision trees) with 25 estimators on
# the iris data. Again, we plot the combined training + test data to see the
# decision boundaries and highlight the test samples.
###############################################################################

from sklearn.ensemble import RandomForestClassifier  # For random forest

# Instantiate the random forest with 25 trees, parallelized with n_jobs=2
forest = RandomForestClassifier(n_estimators=25,
                                random_state=1,
                                n_jobs=2)

# Train the forest
forest.fit(X_train, y_train)

# Plot decision regions
plot_decision_regions(X_combined, y_combined,
                      classifier=forest, test_idx=range(105, 150))
plt.xlabel('Petal length [cm]')
plt.ylabel('Petal width [cm]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

###############################################################################
# %% K-NEAREST NEIGHBORS (KNN)
# Finally, we train a KNN classifier with k=5 using the Euclidean distance
# metric (Minkowski with p=2). We then plot the decision regions.
###############################################################################

from sklearn.neighbors import KNeighborsClassifier  # For KNN

# Create a KNN with 5 neighbors
knn = KNeighborsClassifier(n_neighbors=5, p=2,
                           metric='minkowski')

# Train the KNN on standardized data
knn.fit(X_train_std, y_train)

# Plot decision regions on the combined dataset
plot_decision_regions(X_combined_std, y_combined,
                      classifier=knn, test_idx=range(105, 150))
plt.xlabel('Petal length [standardized]')
plt.ylabel('Petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
