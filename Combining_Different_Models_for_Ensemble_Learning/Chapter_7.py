"""
Combining Different Models for Ensemble Learning - Chapter 7
Luke Bray
"""

from scipy.special import comb
import math

def ensemble_error(n_classifier, error):
    k_start = int(math.ceil(n_classifier / 2.0))

    probs = [
        comb(n_classifier, k) * (error**k) * (1 - error)**(n_classifier - k)
        for k in range(k_start, n_classifier + 1)
    ]

    return sum(probs)

print(ensemble_error(n_classifier=11, error=0.25))

import numpy as np
import matplotlib.pyplot as plt

error_range = np.arange(0.0, 1.01, 0.01)

ens_errors = [
    ensemble_error(n_classifier=11, error=error)
    for error in error_range
]

plt.plot(
    error_range,
    ens_errors,
    label='Ensemble Error',
    linewidth=2
)

plt.plot(
    error_range,
    error_range,
    linestyle='--',
    label='Base error',
    linewidth=2
)

plt.xlabel('Base error')
plt.ylabel('Base/Ensemble Error')
plt.legend(loc='upper left')
plt.grid(alpha=0.5)
plt.show()

# %% Combining classifiers via majority vote

# Implementing a simple majority vote classifier
print(np.argmax(np.bincount([0, 0, 1], weights=[0.2, 0.2, 0.6])))

# Weighted majority based on class probabilities
ex = np.array(
    [[0.9, 0.1,],
     [0.8, 0.2],
     [0.4, 0.6]]
)

p = np.average(ex, axis=0, weights=[0.2, 0.2, 0.6])

print(p)

print(np.argmax(p))

# Majority vote classifier implementation

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone
from sklearn.pipeline import _name_estimators
import operator

class MajorityVoteClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, classifiers, vote='classlabel', weights=None):
        self.classifiers = classifiers
        self._named_classifiers = {
            key: value for key,
            value in _name_estimators(classifiers)
        }
        self.vote = vote
        self.weights = weights

    def fit(self, X, y):
        if self.vote not in ('probability', 'classlabel'):
            raise ValueError(f"vote must be 'probability' "
                             f"or 'classlabel'"
                             f"; got (vote={self.vote})")

        if self.weights and len(self.weights) != len(self.classifiers):
            raise ValueError(f'Number of classifiers and'
                             f' weights must be equal'
                             f'; got {len(self.weights)} weights,'
                             f' {len(self.classifiers)} classifiers')

        self.lablenc_ = LabelEncoder()
        self.lablenc_.fit(y)
        self.classes_ = self.lablenc_.classes_
        self.classifiers_ = []

        for clf in self.classifiers:
            fitted_clf = clone(clf).fit(X, self.lablenc_.transform(y))
            self.classifiers_.append(fitted_clf)

        return self

    def predict(self, X):
        if self.vote == 'probability':
            maj_vote = np.argmax(self.predict_proba(X), axis=1)
        else:
            predictions = np.asarray([
                clf.predict(X) for clf in self.classifiers_
            ]).T

            maj_vote = np.apply_along_axis(
                lambda x: np.argmax(
                    np.bincount(x, weights=self.weights)
                ),
                axis=1,
                arr=predictions
            )

        maj_vote = self.lablenc_.inverse_transform(maj_vote)

        return maj_vote

    def predict_proba(self, X):
        probas = np.asarray([clf.predict_proba(X)
                             for clf in self.classifiers_])

        avg_proba = np.average(probas, axis=0, weights=self.weights)

        return avg_proba

    def get_params(self, deep=True):
        if not deep:
            return super().get_params(deep=False)
        else:
            out = self.named_classifiers.copy()
            for name, step in self.named_classifiers.items():
                for key, value in step.get_params(deep=True).items():
                    out[f'{name}__{key}'] = value

            return out

# %% Using the majority voting principle to make predictions

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

iris = datasets.load_iris()
X, y = iris.data[50:, [1, 2]], iris.target[50:]
le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.5,
                                                    random_state=1,
                                                    stratify=y)

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

clf1 = LogisticRegression(penalty='12',
                          C=0.001,
                          solver='lbfgs',
                          random_state=1)

clf2 = DecisionTreeClassifier(max_depth=1,
                              criterion='entropy',
                              random_state=0)

clf3 = KNeighborsClassifier(n_neighbors=1,
                            p=2,
                            metric='minkowski')

pipe1 = Pipeline([['sc', StandardScaler()],
                  ['clf', clf1]])

pipe3 = Pipeline([['sc', StandardScaler()],
                  ['clf', clf3]])

clf_labels = ['Logistic Regression', 'Decision Tree', 'KNN']

print('10-fold cross validation:\n')

for clf, label in zip([pipe1, clf2, pipe3], clf_labels):
    scores = cross_val_score(estimator=clf,
                             X=X_train,
                             y=y_train,
                             cv=10,
                             scoring='roc_auc')
    print(f'ROC AUC: {scores.mean():.2f} '
          f'(+/- {scores.std():.2f}) [{label}]')


mv_clf = MajorityVoteClassifier(classifiers=[pipe1, clf2, pipe3])

clf_labels += ['Majority voting']

all_clf = [pipe1, clf2, pipe3, mv_clf]

for clf, label in zip(all_clf, clf_labels):
    scores = cross_val_score(estimator=clf,
                             X=X_train,
                             y=y_train,
                             cv=10,
                             scoring='roc_auc')

    print(f'ROC AUC: {scores.mean():.2f} '
          f'(+/- {scores.std():.2f}) [{label}]')


# %% Bagging - building an ensemble of classifiers from bootstrap samples

import pandas as pd

# Define the path to the dataset file
data_path = 'machine-learning-databases/wine.data'

# Load the dataset (the file has no header)
df_wine = pd.read_csv(data_path, header=None)

# Assign column names to the DataFrame
df_wine.columns = [
    'Class label', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash',
    'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols',
    'Proanthocyanins', 'Color intensity', 'Hue',
    'OD280/OD315 of diluted wines', 'Proline'
]

# Drop rows where the class label is 1
df_wine = df_wine[df_wine['Class label'] != 1]

# Extract the target variable (class labels)
y = df_wine['Class label'].values

# Extract the features: 'Alcohol' and 'OD280/OD315 of diluted wines'
X = df_wine[['Alcohol', 'OD280/OD315 of diluted wines']].values

# Optional: print the first few rows of the DataFrame to verify
print(df_wine.head())


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

le = LabelEncoder()
y = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=1,
                                                    stratify=y)

from sklearn.ensemble import BaggingClassifier

tree = DecisionTreeClassifier(criterion='entropy',
                              random_state=1,
                              max_depth=None)

bag = BaggingClassifier(base_estimator=tree,
                        n_estimators=500,
                        max_samples=1,
                        max_features=1,
                        bootstrap=True,
                        bootstrap_features=False,
                        n_jobs=1,
                        random_state=1)

from sklearn.metrics import accuracy_score

tree = tree.fit(X_train, y_train)

y_train_pred = tree.predict(X_train)
y_test_pred = tree.predict(X_test)

tree_train = accuracy_score(y_train, y_train_pred)
tree_test = accuracy_score(y_test, y_test_pred)

print(f'Decision tree train/test accuracies '
      f'{tree_train:.3f}/{tree_test:.3f}')

# High discrepancy between train accuracy and test accuracy infers that the model is extremely overfit

bag = bag.fit(X_train, y_train)

y_train_pred = bag.predict(X_train)
y_test_pred = bag.predict(X_test)

bag_train = accuracy_score(y_train, y_train_pred)
bag_test = accuracy_score(y_test, y_test_pred)

print(f'Bagging train/test accuracies '
      f'{bag_train:.3f}/{bag_test:.3f}')


# Define the boundaries of the grid for plotting decision regions
x_min = X_train[:, 0].min() - 1
x_max = X_train[:, 0].max() + 1
y_min = X_train[:, 1].min() - 1
y_max = X_train[:, 1].max() + 1

# Create a mesh grid using a step size of 0.1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# Create subplots: 1 row, 2 columns; share the x-axis in the columns and the y-axis in the rows
fig, axarr = plt.subplots(nrows=1, ncols=2, sharex='col', sharey='row', figsize=(8, 3))

# Iterate over the classifiers and plot decision regions along with training data
for idx, clf, title in zip([0, 1], [tree, bag], ['Decision tree', 'Bagging']):
    # Fit the classifier on the training data
    clf.fit(X_train, y_train)

    # Predict the class labels for each point in the grid
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the decision boundary using contour fill
    axarr[idx].contourf(xx, yy, Z, alpha=0.3)

    # Plot training data points: class 0 with blue triangles and class 1 with green circles
    axarr[idx].scatter(X_train[y_train == 0, 0],
                       X_train[y_train == 0, 1],
                       c='blue', marker='^', label='Class 0')
    axarr[idx].scatter(X_train[y_train == 1, 0],
                       X_train[y_train == 1, 1],
                       c='green', marker='o', label='Class 1')

    # Set the title for the subplot
    axarr[idx].set_title(title)

# Set the y-axis label for the left subplot
axarr[0].set_ylabel('OD280/OD315 of diluted wines', fontsize=12)

# Adjust layout to prevent overlapping elements
plt.tight_layout()

# Add a shared x-axis label using text
plt.text(0, -0.2, s='Alcohol', ha='center', va='center', fontsize=12, transform=fig.transFigure)

# Display the plot
plt.show()


# %% Boosting





