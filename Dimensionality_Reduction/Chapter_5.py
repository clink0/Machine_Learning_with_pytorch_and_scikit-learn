################################################################################
# Title: Compressing Data via Dimensionality Reduction - Chapter 5
# Author: Luke Bray
# Date: February 6, 2025
#
# Explanation Summary:
# 
# 1. Principal Component Analysis (PCA) Steps
#    - Standardizes the data.
#    - Computes the covariance matrix, eigenvalues, and eigenvectors.
#    - Sorts and selects principal components.
#    - Plots explained variance and projects onto principal components for visualization.
#
# 2. Plot Decision Regions after PCA
#    - Applies PCA transformation to training and test data.
#    - Fits a logistic regression model on the projected data.
#    - Plots decision regions on both training and test sets in the new 2D space.
#
# 3. Assessing Feature Contributions to PCA
#    - Shows how to compute loadings (i.e., how each feature contributes to a principal component).
#    - Demonstrates manual computation and scikit-learn’s built-in approach.
#
# 4. Supervised Data Compression via Linear Discriminant Analysis (LDA)
#    - Constructs between-class and within-class scatter matrices from the standardized data.
#    - Computes eigenpairs, sorts them, and selects linear discriminants (LDs) for maximum class separation.
#
# 5. Projecting Examples on New Feature Space (LDA)
#    - Projects data onto the top two LDs and visualizes class separation.
#
# 6. LDA via scikit-learn
#    - Uses scikit-learn’s LDA to reduce dimensionality to 2D.
#    - Trains and visualizes a logistic regression model in LD space on both train and test data.
#
# 7. Nonlinear Dimensionality Reduction and Visualization (t-SNE)
#    - Loads the handwritten digits dataset.
#    - Applies t-SNE to embed the digits data in 2D for visualization.
################################################################################

"""
Overview:
Principal component analysis for unsupervised data compression
Linear discriminant analysis as a supervised dimensionality reduction technique for maximizing class seperability
Nonlinear dimensionality reduction techniques and t-distributed stochastic neighbor embedding for data visualization
"""


# %% First four steps of the Principal Component Analysis (PCA)
# 1. Standardize data
# 2. Construct covariance matrix
# 3. Obtain eigenvalues and eigenvectors
# 4. Sort eigenvalues in decreasing order to select principal components

import pandas as pd  # For data manipulation and reading CSV data
import matplotlib.pyplot as plt  # For plotting figures
import numpy as np  # For numerical computations

# Read the Wine dataset from UCI repository
df_wine = pd.read_csv(
    'https://archive.ics.uci.edu/ml/'
    'machine-learning-databases/wine/wine.data',
    header=None
)

# Split features and target labels, dividing data into training and test sets
from sklearn.model_selection import train_test_split
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.3,        # 30% for test set
    stratify=y,           # preserve class distribution
    random_state=0        # for reproducibility
)

# Standardize the features
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()                   # Create a StandardScaler object
X_train_std = sc.fit_transform(X_train) # Fit and transform the training data
X_test_std = sc.transform(X_test)       # Transform the test data using same parameters

# Compute the covariance matrix on the transposed standardized data
cov_mat = np.cov(X_train_std.T)         # Covariance matrix of features
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)  # Eigen decomposition
print('\nEigenvalues \n', eigen_vals)

# Calculate variance explained by each eigenvalue, then cumulative sum for plotting
tot = sum(eigen_vals)                               # Sum of all eigenvalues
var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

# Plot the variance explained ratios
plt.bar(range(1, 14), var_exp, align='center', 
        label='Individual explained variance')
plt.step(range(1, 14), cum_var_exp, where='mid', 
         label='Cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

# Sort eigenvalue-eigenvector pairs by decreasing eigenvalues
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) 
               for i in range(len(eigen_vals))]
eigen_pairs.sort(key=lambda k: k[0], reverse=True)

# Select top two eigenvectors for a 2D projection
w = np.hstack((
    eigen_pairs[0][1][:, np.newaxis],
    eigen_pairs[1][1][:, np.newaxis]
))
print('Matrix W:\n', w)

# Project the first standardized training sample onto the principal components
print(X_train_std[0].dot(w))

# Create 2D projection of the entire training set
X_train_pca = X_train_std.dot(w)

# Plot the projected samples by class
colors = ['r', 'b', 'g']    # Red, blue, green
markers = ['o', 's', '^']   # Circle, square, triangle
for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(
        X_train_pca[y_train == l, 0],
        X_train_pca[y_train == l, 1],
        c=c, label=f'Class{l}', marker=m
    )
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()


# %% Plot decision regions after PCA
# Here we use scikit-learn's PCA to transform the data into 2D and fit a 
# logistic regression model. We'll then plot the decision boundaries for 
# both the training and test sets in the PCA-reduced space.

from plot_decision_regions_script import plot_decision_regions  # Custom plotting function
from sklearn.linear_model import LogisticRegression             # For classification
from sklearn.decomposition import PCA                           # PCA transformation

pca = PCA(n_components=2)                                       # Create PCA object with 2 PCs
lr = LogisticRegression(multi_class='ovr', random_state=1, solver='lbfgs')  # Logistic regression

# Transform both training and test data into 2D PCA space
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)

# Fit logistic regression on the 2D PCA representation
lr.fit(X_train_pca, y_train)

# Plot decision regions for the training set in PCA space
plot_decision_regions(X_train_pca, y_train, classifier=lr)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()

# Plot decision regions for the test set in PCA space
plot_decision_regions(X_test_pca, y_test, classifier=lr)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()

# Fit a PCA with all components to see explained variance ratio
pca = PCA(n_components=None)
X_train_pca = pca.fit_transform(X_train_std)
print(pca.explained_variance_ratio_)


# %% Assessing feature contributions (loadings)
# Loadings indicate how much each original feature contributes to a principal
# component. We'll demonstrate the manual computation via eigenvectors and 
# scikit-learn's built-in approach.

# Calculate loadings manually: loadings = eigen_vecs * sqrt(eigen_vals)
loadings = eigen_vecs * np.sqrt(eigen_vals)

# Plot loadings for the first principal component
fig, ax = plt.subplots()
ax.bar(range(13), loadings[:, 0], align='center')
ax.set_ylabel('Loadings for PC 1')
ax.set_xticks(range(13))
ax.set_xticklabels(df_wine.columns[1:], rotation=90)
plt.ylim([1,1])  # This might be intentionally set or can be adjusted
plt.tight_layout()
plt.show()

# Alternative method using scikit-learn's PCA (components_ and explained_variance_)
sklearn_loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

fig, ax = plt.subplots()
ax.bar(range(13), sklearn_loadings[:, 0], align='center')
ax.set_ylabel('Loadings for PC 1')
ax.set_xticks(range(13))
ax.set_xticklabels(df_wine.columns[1:], rotation=90)
plt.ylim([-1, 1])
plt.tight_layout()
plt.show()


# %% Supervised data compression via linear discriminant analysis (LDA)
# 1. Standardize the d-dimensional dataset.
# 2. For each class, compute the d-dimensional mean vector.
# 3. Construct the between-class scatter matrix S_B and the within-class scatter matrix S_W.
# 4. Compute the eigenvectors and eigenvalues of S_W^-1 * S_B.
# 5. Sort the eigenvalues by decreasing order and select the top k eigenvectors.
# 6. Form a projection matrix W from those eigenvectors.
# 7. Transform data X via X * W to get a new feature subspace.

# Compute the mean vectors per class
np.set_printoptions(precision=4)  # For nicer numeric output
mean_vecs = []

for label in range(1, 4):
    # Compute class mean for each dimension
    mean_vecs.append(np.mean(X_train_std[y_train == label], axis=0))
    print(f'MV {label}: {mean_vecs[label-1]}\n')

# We have 13 features in the Wine dataset
d = 13

# Construct the within-class scatter matrix (unscaled version)
S_W = np.zeros((d, d))
for label, mv in zip(range(1, 4), mean_vecs):
    class_scatter = np.zeros((d, d))
    # Sum over all samples from each class
    for row in X_train_std[y_train == label]:
        row, mv = row.reshape(d, 1), mv.reshape(d, 1)
        class_scatter += (row - mv).dot((row - mv).T)
    S_W += class_scatter
print('Within-class scatter matrix: '
      f'{S_W.shape[0]}x{S_W.shape[1]}')

# Check class distribution to see if classes are balanced
print('Class label distribution: ', np.bincount(y_train)[1:])

# We can scale the within-class scatter by using the class covariance
S_W = np.zeros((d, d))
for label, mv in zip(range(1, 4), mean_vecs):
    # Covariance for each class subset
    class_scatter = np.cov(X_train_std[y_train == label].T)
    S_W += class_scatter
print('Scaled within-class scatter matrix: '
      f'{S_W.shape[0]}x{S_W.shape[1]}')

# Compute overall mean (across all classes)
mean_overall = np.mean(X_train_std, axis=0)
mean_overall = mean_overall.reshape(d, 1)

# Construct the between-class scatter matrix
S_B = np.zeros((d, d))
for i, mean_vec in enumerate(mean_vecs):
    n = X_train_std[y_train == i + 1, :].shape[0]  # number of samples in class i
    mean_vec = mean_vec.reshape(d, 1)
    S_B += n * (mean_vec - mean_overall).dot((mean_vec - mean_overall).T)
print('Between class scatter matrix: '
      f'{S_B.shape[0]}x{S_B.shape[1]}')

# Solve generalized eigenvalue problem for S_W^-1 * S_B
eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
eigen_pairs = [
    (np.abs(eigen_vals[i]), eigen_vecs[:, i])
    for i in range(len(eigen_vals))
]
eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)
print('Eigenvalues in descending order:\n')
for eigen_val in eigen_pairs:
    print(eigen_val[0])

# Plot individual and cumulative "discriminability" ratio
tot = sum(eigen_vals.real)
discr = [(i / tot) for i in sorted(eigen_vals.real, reverse=True)]
cum_discr = np.cumsum(discr)

plt.bar(range(1, 14), discr, align='center',
        label='Individual discriminability')
plt.step(range(1, 14), cum_discr, where='mid',
         label='Cumulative discriminability')
plt.ylabel('"Discriminbility" ratio')
plt.xlabel('Linear Discriminants')
plt.ylim([-0.1, 1.1])
plt.legend(loc='best')
plt.tight_layout()
plt.show()

# Create the transformation matrix W from the top 2 eigenvectors
w = np.hstack((
    eigen_pairs[0][1][:, np.newaxis].real,
    eigen_pairs[1][1][:, np.newaxis].real
))
print('Matrix W:\n', w)


# %% Projecting examples on the new feature space (LDA)
# Here we manually transform the training data with our selected eigenvectors.
X_train_lda = X_train_std.dot(w)

# Plot the data in the new 2D LDA space
colors = ['r', 'b', 'g']    # Red, blue, green
markers = ['o', 's', '^']   # Circle, square, triangle

for l, c, m in zip(np.unique(y_train), colors, markers):
    # Multiply the second LD by -1 if needed for orientation
    plt.scatter(
        X_train_lda[y_train == l, 0],
        X_train_lda[y_train == l, 1] * (-1),
        c=c, label=f'Class {l}', marker=m
    )

plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()


# %% LDA via scikit-learn
# We use scikit-learn’s LinearDiscriminantAnalysis to directly reduce the data 
# to 2D, then train a logistic regression model for classification.

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

lda = LDA(n_components=2)  # Keep 2 linear discriminants
X_train_lda = lda.fit_transform(X_train_std, y_train)

lr = LogisticRegression(multi_class='ovr', random_state=1, solver='lbfgs')
lr.fit(X_train_lda, y_train)

# Plot decision regions for the training set in LDA space
plot_decision_regions(X_train_lda, y_train, classifier=lr)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()

# Evaluate the same model on test data (transformed via LDA)
X_test_lda = lda.transform(X_test_std)
plot_decision_regions(X_test_lda, y_test, classifier=lr)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()


# %% Nonlinear dimensionality reduction and visualization (t-SNE)
# We load the digits dataset, then apply t-SNE to embed the data in 2D space
# for visual exploration of the digit clusters.

from sklearn.datasets import load_digits  # Built-in digits dataset in scikit-learn
digits = load_digits()                    # Load the dataset of 8x8 images

fig, ax = plt.subplots(1, 4)             # Create a row of 4 subplots

# Display the first 4 images in grayscale
for i in range(4):
    ax[i].imshow(digits.images[i], cmap='Greys')

plt.show()

# digits.data has shape [n_samples, n_features], i.e., flatten 8x8 into 64 features
print(digits.data.shape)

y_digits = digits.target  # The digit labels (0 through 9)
X_digits = digits.data    # Flattened pixel intensities

from sklearn.manifold import TSNE        # t-SNE for manifold learning
tsne = TSNE(n_components=2, init='pca', random_state=123)
X_digits_tsne = tsne.fit_transform(X_digits)

# Function to plot the 2D embedded points, color-coded by digit
import matplotlib.patheffects as PathEffects
def plot_projection(x, colors):
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    for i in range(10):
        plt.scatter(x[colors == i, 0],
                    x[colors == i, 1])
    for i in range(10):
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        # PathEffects to create a white outline around the text for clarity
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()
        ])

# Plot the t-SNE projection of the digits
plot_projection(X_digits_tsne, y_digits)
plt.show()
