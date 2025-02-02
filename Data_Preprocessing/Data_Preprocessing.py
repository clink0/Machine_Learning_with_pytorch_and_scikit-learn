################################################################################
# Title: Chapter 4: Building Good Training Datasets - Data Preprocessing
# Author: Luke Bray
# Date: February 2, 2025
#
# Explanation Summary:
# 
# 1. Create DataFrame and Find Null Values in Each Column:
#    - Reads CSV data into a DataFrame, identifies NaNs, and demonstrates
#      various ways to drop or keep rows/columns with missing data.
#
# 2. Imputing Missing Values using Scikit-Learn:
#    - Uses SimpleImputer to replace NaNs in the DataFrame with the mean
#      of each column.
#
# 3. Imputing Missing Values using pandas:
#    - Demonstrates pandas' built-in .fillna() function to fill missing
#      values with column means.
#
# 4. Categorical Data Encoding with pandas:
#    - Shows how to encode ordinal features (e.g., size) via a dictionary
#      mapping and how to invert that mapping.
#
# 5. Encoding Class Labels:
#    - Encodes class labels into integer indices both via a custom
#      dictionary mapping and LabelEncoder from scikit-learn. Also
#      demonstrates one-hot encoding for nominal features like color.
#
# 6. Partitioning a Dataset into Train and Test:
#    - Uses train_test_split to separate features and labels into training
#      and test sets. Ensures class distributions remain similar by
#      stratification.
#
# 7. Feature Scaling:
#    - Illustrates normalization (MinMaxScaler) and standardization (StandardScaler),
#      discussing the difference and when each might be used.
#
# 8. Regularization:
#    - Shows how to apply L1 regularization in logistic regression to help
#      with feature selection. Prints training/test accuracy and model
#      coefficients.
#
# 9. Plot Regularization Strength and Path:
#    - Iterates over different C values (inverse regularization strength)
#      in a logistic regression model, tracking how the weight coefficients
#      change. Plots these coefficients to illustrate how regularization
#      affects them.
#
# 10. Sequential Backward Selection (SBS):
#     - Implements a custom SBS class to iteratively remove features,
#       evaluating performance with KNN at each step. Plots accuracy vs.
#       the number of retained features and identifies an optimal subset.
#
# 11. Assessing Feature Importance with Random Forest:
#     - Trains a random forest to measure the relative importance of features,
#       sorts them, and plots the feature importances for comparison.
################################################################################

# %% Create dataframe and find null values in each column
# This section demonstrates how to read CSV data into a pandas DataFrame, identify
# missing (NaN) values, and remove rows or columns with NaNs using multiple strategies.

import pandas as pd  # For data manipulation and analysis
from io import StringIO  # Allows treating strings as file-like objects

from pandas.core.interchange.from_dataframe import categorical_column_to_series
# (Likely an internal import - not used directly here but left as is.)

# Sample CSV data with missing values represented as empty commas
csv_data = \
'''A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
10.0,11.0,12.0,'''

# Reading CSV data from a string, converting it into a DataFrame
df = pd.read_csv(StringIO(csv_data))

# Print the newly created DataFrame
print(df)

# Print the sum of null values in each column
print(df.isnull().sum())

# Remove rows containing NaN values (not assigned to a variable -> ephemeral)
df.dropna(axis=0)

# Remove columns containing NaN values (again, ephemeral if not assigned)
df.dropna(axis=1)

# Only drop rows where all columns are NaN
df.dropna(how='all')

# Drop rows that have fewer than 4 real (non-NaN) values
df.dropna(thresh=4)

# Only drop rows where NaN appear in specific columns (in this case, column 'C')
df.dropna(subset=['C'])


# %% Imputing missing values using scikit learn
# Instead of dropping rows/columns, we can replace missing values with the mean,
# median, or another strategy. Here we use SimpleImputer from scikit-learn to
# replace missing values with the mean of each column.

from sklearn.impute import SimpleImputer  # For imputation of missing data
import numpy as np  # For numerical operations

# Create a SimpleImputer to fill NaNs with the column mean
imr = SimpleImputer(missing_values=np.nan, strategy='mean')

# Fit the imputer on the DataFrame's values
imr = imr.fit(df.values)

# Transform the DataFrame by replacing NaNs with the mean
imputed_data = imr.transform(df.values)

# Print the transformed data array
print(imputed_data)


# %% Imputing missing values using pandas
# Demonstrates pandas' fillna() function, which can replace NaNs with a single
# value or the result of an aggregation (e.g., the mean of each column).

# Print the DataFrame after replacing NaNs with the mean of each column
print(df.fillna(df.mean()))


# %% Categorical data encoding with pandas
# We create a new DataFrame with categorical features (color, size) and a target
# classlabel column. Then we map the "size" feature from text to integers.

df = pd.DataFrame([
           ['green', 'M', 10.1, 'class2'],
           ['red', 'L', 13.5, 'class1'],
           ['blue', 'XL', 15.3, 'class2']])

# Assign column names
df.columns = ['color', 'size', 'price', 'classlabel']
print(df)

# Define an ordered mapping for size
size_mapping = {'XL': 3, 'L': 2, 'M': 1}

# Map the size column using the predefined dictionary
df['size'] = df['size'].map(size_mapping)
print(df)

# Invert the mapping
inv_size_mapping = {v: k for k, v in size_mapping.items()}

# Use the inverted mapping to get back original text values
df['size'].map(inv_size_mapping)


# %% Encoding class labels
# We encode the class labels (classlabel column) into numeric integers for use
# in machine learning algorithms. Demonstrate both manual dict mapping and
# scikit-learn's LabelEncoder.

# Create a dictionary to map each class label to an integer
class_mapping = {label: idx for idx, label in enumerate(np.unique(df['classlabel']))}
print(class_mapping)

# Transform the class labels using our custom dictionary
df['classlabel'] = df['classlabel'].map(class_mapping)
print(df)

# Invert the class label encoding
inv_class_mapping = {v: k for k, v in class_mapping.items()}
df['classlabel'] = df['classlabel'].map(inv_class_mapping)
print(df)

# Alternative using scikit-learn LabelEncoder
from sklearn.preprocessing import LabelEncoder
class_le = LabelEncoder()

# Convert class labels to integer form
y = class_le.fit_transform(df['classlabel'].values)
print(y)

# Invert the label encoding back to original class labels
class_le.inverse_transform(y)

# Encode nominal (color) features
X = df[['color', 'size', 'price']].values
color_le = LabelEncoder()
X[:, 0] = color_le.fit_transform(X[:, 0])
print(X)

# One-hot encoding for the first column (color)
from sklearn.preprocessing import OneHotEncoder
X = df[['color', 'size', 'price']].values
color_ohe = OneHotEncoder(categories='auto', drop='first')
# Fit-transform reshaped color column and convert to array
color_ohe.fit_transform(X[:, 0].reshape(-1, 1)).toarray()

# Using ColumnTransformer for multiple transformations
from sklearn.compose import ColumnTransformer
X = df[['color', 'size', 'price']].values
c_transf = ColumnTransformer([
    ('onehot', OneHotEncoder(), [0]),    # Apply OneHotEncoder to the first column
    ('nothing', 'passthrough', [1, 2])   # Pass the other columns unchanged
])
print(c_transf.fit_transform(X).astype(float))

# Alternative (simpler) implementation using pandas.get_dummies()
print(pd.get_dummies(df[['price', 'color', 'size']]))

# drop_first=True to avoid dummy variable trap (redundant column)
print(pd.get_dummies(df[['price', 'color', 'size']], drop_first=True))


# %% Partitioning a dataset into test and training sub-datasets
# Loads the Wine dataset from UCI repository, shows the class labels, and
# separates them into train and test sets for model training and evaluation.

df_wine = pd.read_csv('https://archive.ics.uci.edu/'
                      'ml/machine-learning-databases/'
                      'wine/wine.data', header=None)

# Name the columns for clarity
df_wine.columns = ['Class label', 'Alcohol',
                   'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium',
                   'Total phenols', 'Flavanoids',
                   'Nonflavanoid phenols',
                   'Proanthocyanins',
                   'Color intensity', 'Hue',
                   'OD280/OD315 of diluted wines',
                   'Proline']

# Print the unique class labels
print('Class labels', np.unique(df_wine['Class label']))

# Preview the first few rows
print(df_wine.head())

# Use train_test_split to create train/test sets (stratify ensures balanced classes)
from sklearn.model_selection import train_test_split

# Separate features (X) and labels (y)
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

# 70% training, 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3, 
                                                    random_state=0, 
                                                    stratify=y)


# %% Feature scaling
# Standardization and MinMaxScaler are demonstrated. The code also compares
# standardization vs. normalization with a simple example array.

from sklearn.preprocessing import MinMaxScaler  # For normalization
mms = MinMaxScaler()

# Fit the normalization scaler on X_train and transform both train and test
X_train_norm = mms.fit_transform(X_train)
X_test_norm = mms.transform(X_test)

# StandardScaler is more common; often more robust to outliers.
from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()

# Fit on the training set and transform
X_train_std = stdsc.fit_transform(X_train)
# Transform the test set using the same parameters
X_test_std = stdsc.transform(X_test)

# Compare standardization vs. normalization on a small example
ex = np.array([0, 1, 2, 3, 4, 5])
print('standardized:', (ex - ex.mean()) / ex.std())
print('normalized:', (ex - ex.min()) / (ex.max() - ex.min()))


# %% Regularization (a method of feature selection)
# Demonstrates L1 regularization in logistic regression. The coefficient
# array can be examined to see which features are pushed toward zero.

from sklearn.linear_model import LogisticRegression

# Logistic regression with L1 penalty (liblinear solver required)
lr = LogisticRegression(penalty='l1', C=1.0, solver='liblinear', multi_class='ovr')
lr.fit(X_train_std, y_train)

# Print training and test accuracy
print('Training accuracy:', lr.score(X_train_std, y_train))
print('Test accuracy:', lr.score(X_test_std, y_test))

# Check intercept for each class
print(lr.intercept_)

# Check weight array (coefficients) for each class
print(lr.coef_)


# %% Plot regularization strength and path
# We vary the inverse regularization strength parameter C for logistic
# regression. We store and plot how each feature's weight changes as C changes.

import matplotlib.pyplot as plt  # For plotting

fig = plt.figure()
ax = plt.subplot(111)

# Assign different colors to different features
colors = ['blue', 'green', 'red', 'cyan',
          'magenta', 'yellow', 'black',
          'pink', 'lightgreen', 'lightblue',
          'gray', 'indigo', 'orange']

weights, params = [], []

# Evaluate powers of 10 for C, from 10^-4 to 10^5
for c in np.arange(-4., 6.):
    lr = LogisticRegression(penalty='l1', C=10.**c,
                            solver='liblinear',
                            multi_class='ovr', random_state=0)
    lr.fit(X_train_std, y_train)
    # We'll look at the coefficients for class index 1
    weights.append(lr.coef_[1])
    params.append(10**c)

# Convert weights to a NumPy array for easy slicing
weights = np.array(weights)

# Plot each feature's coefficient path
for column, color in zip(range(weights.shape[1]), colors):
    plt.plot(params, weights[:, column],
             label=df_wine.columns[column + 1],
             color=color)

# Draw a horizontal line at 0 to highlight sign changes
plt.axhline(0, color='black', linestyle='--', linewidth=3)

# Limit the x-axis range
plt.xlim([10**(-5), 10**5])

# Label the plot
plt.ylabel('Weight coefficient')
plt.xlabel('C (inverse regularization strength)')
plt.xscale('log')

# Show legend outside the top-right region
plt.legend(loc='upper left')
ax.legend(loc='upper center',
          bbox_to_anchor=(1.38, 1.03),
          ncol=1, fancybox=True)

# Show the plot
plt.show()


# %% Sequential feature selection (SBS)
# Defines a custom SBS class that repeatedly removes features, evaluating
# performance (accuracy) at each step using a given estimator. Here, we
# demonstrate it with a KNN classifier.

from sklearn.base import clone  # For creating a fresh copy of an estimator
from itertools import combinations  # For iterating over subsets
from sklearn.metrics import accuracy_score  # For scoring predicted labels
from sklearn.model_selection import train_test_split  # For splitting

class SBS:
    def __init__(self, estimator, k_features, scoring=accuracy_score, test_size=0.25, random_state=1):
        self.scoring = scoring                 # The scoring function (accuracy)
        self.estimator = clone(estimator)      # A clone of the model we want to wrap
        self.k_features = k_features           # Target number of features
        self.test_size = test_size             # Fraction of data for testing
        self.random_state = random_state       # Seed for reproducibility

    def fit(self, X, y):
        # Split data for SBS evaluation
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=self.test_size,
                                                            random_state=self.random_state)
        # Starting dimension = total number of features
        dim = X_train.shape[1]

        # Start with all feature indices
        self.indices_ = tuple(range(dim))

        # Initialize subsets_ with the full set of features
        self.subsets_ = [self.indices_]

        # Evaluate score with all features
        score = self._calc_score(X_train, y_train, X_test, y_test, self.indices_)
        self.scores_ = [score]

        # Iteratively remove features until we reach k_features
        while dim > self.k_features:
            scores = []
            subsets = []

            # Generate all possible feature subsets of size (dim - 1)
            for p in combinations(self.indices_, r=dim - 1):
                score = self._calc_score(X_train, y_train, X_test, y_test, p)
                scores.append(score)
                subsets.append(p)

            # Choose the subset of size (dim - 1) that yields the best score
            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            dim -= 1

            # Append the best score for this iteration
            self.scores_.append(scores[best])

        # Final (best) score
        self.k_score_ = self.scores_[-1]
        return self

    def transform(self, X):
        # Return the dataset containing only the chosen feature indices
        return X[:, self.indices_]

    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        # Fit model using only specific feature indices
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)
        return score


import matplotlib.pyplot as plt  # For plotting
from sklearn.neighbors import KNeighborsClassifier  # KNN classifier

# Create a KNN classifier with k=5
knn = KNeighborsClassifier(n_neighbors=5)

# Instantiate the SBS with the KNN and target k_features=1
sbs = SBS(knn, k_features=1)

# Run the SBS search on the standardized training set
sbs.fit(X_train_std, y_train)

# Plot the accuracy vs. the number of features
k_feat = [len(k) for k in sbs.subsets_]
plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylim([0.7, 1.02])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.tight_layout()
plt.show()

# Example: The smallest feature subset (k = 3) was found around iteration 10
k3 = list(sbs.subsets_[10])
print(df_wine.columns[1:][k3])

# Evaluate performance of KNN classifier on full feature set
knn.fit(X_train_std, y_train)
print('Training accuracy:', knn.score(X_train_std, y_train))
print('Test accuracy:', knn.score(X_test_std, y_test))

# Evaluate performance on the three-feature subset
knn.fit(X_train_std[:, k3], y_train)
print('Training accuracy:', knn.score(X_train_std[:, k3], y_train))
print('Test accuracy:', knn.score(X_test_std[:, k3], y_test))


# %% Assessing feature importance with random forests
# We train a random forest on the Wine data and examine the importance scores
# for each feature. Then, we plot those importance scores in descending order.

from sklearn.ensemble import RandomForestClassifier

# Store the feature labels for display
feat_labels = df_wine.columns[1:]

# Create a random forest classifier with 500 estimators
forest = RandomForestClassifier(n_estimators=500, random_state=1)

# Fit the forest to the original (unscaled) training data
forest.fit(X_train, y_train)

# Retrieve the feature importances
importances = forest.feature_importances_

# Sort the importance indices in descending order
indices = np.argsort(importances)[::-1]

# Print the feature rankings
for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30,
                            feat_labels[indices[f]],
                            importances[indices[f]]))

# Plot the feature importance scores
plt.title('Feature importance')
plt.bar(range(X_train.shape[1]), importances[indices], align='center')
plt.xticks(range(X_train.shape[1]), feat_labels[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()