"""
Chapter 9: Predicting Continuous Target Variables with Regression Analysis
Luke Bray
February 13, 2025
"""

################################################################################
# %% Importing dataset
# This section imports the Ames Housing dataset, selects a subset of columns,
# converts categorical variables to numeric form, checks for missing values,
# and drops any rows with missing data.
################################################################################

import pandas as pd  # For data manipulation and file I/O

# Define the columns to be used from the dataset
columns = ['Overall Qual', 'Overall Cond', 'Gr Liv Area',
           'Central Air', 'Total Bsmt SF', 'SalePrice']

# Read the dataset from the URL (tab-separated values) and only use specified columns
df = pd.read_csv('http://jse.amstat.org/v19n3/decock/AmesHousing.txt',
                 sep='\t', usecols=columns)

# Print the first few rows of the DataFrame to inspect the data
print(df.head())

# Print the shape of the DataFrame (number of rows and columns)
print(df.shape)

# Map the 'Central Air' column from categorical ('N'/'Y') to numerical (0/1)
df['Central Air'] = df['Central Air'].map({'N': 0, 'Y': 1})

# Check for missing values in each column
print(df.isnull().sum())

# Drop any rows that contain missing values
df = df.dropna(axis=0)
# Verify that no missing values remain
print(df.isnull().sum())


################################################################################
# %% Visualizing the important characteristics of a dataset
# This section creates a scatterplot matrix and a heatmap of the correlation matrix
# to visualize relationships between features in the dataset.
################################################################################

import matplotlib.pyplot as plt  # For plotting
from mlxtend.plotting import scatterplotmatrix  # For scatterplot matrix visualization

# Create a scatterplot matrix of the dataset values with specified figure size and transparency
scatterplotmatrix(df.values, figsize=(12, 10), names=df.columns, alpha=0.5)
plt.tight_layout()
plt.show()

# Compute the correlation matrix (Pearson correlation) from the transposed data
import numpy as np
from mlxtend.plotting import heatmap  # For heatmap visualization

cm = np.corrcoef(df.values.T)
# Plot the correlation matrix as a heatmap with row and column names
hm = heatmap(cm, row_names=df.columns, column_names=df.columns)
plt.tight_layout()
plt.show()


################################################################################
# %% Implementing an ordinary least squares linear regression model
# This section implements linear regression using gradient descent.
# A custom class LinearRegressionGD is defined, trained on standardized data,
# and its performance is visualized through loss plots and regression line plots.
################################################################################

# Solving regression parameters with gradient descent
class LinearRegressionGD:
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta                  # Learning rate
        self.n_iter = n_iter            # Number of iterations (epochs)
        self.random_state = random_state  # Seed for reproducibility

    def fit(self, X, y):
        # Initialize random number generator and weights (coefficients) with small random numbers
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.array([0.])        # Initialize bias term
        self.losses_ = []               # List to store the mean squared error at each epoch

        # Iterate over the specified number of iterations
        for i in range(self.n_iter):
            output = self.net_input(X)   # Compute the predicted values
            errors = (y - output)        # Compute the error vector
            # Update weights and bias using gradient descent update rule
            self.w_ += self.eta * 2.0 * X.T.dot(errors) / X.shape[0]
            self.b_ += self.eta * 2.0 * errors.mean()
            # Compute mean squared error for the current iteration and store it
            loss = (errors**2).mean()
            self.losses_.append(loss)
        return self

    def net_input(self, X):
        # Compute the net input (linear combination of features and weights plus bias)
        return np.dot(X, self.w_) + self.b_

    def predict(self, X):
        # Predict the target values (for regression, this is just the net input)
        return self.net_input(X)

# Extract the feature 'Gr Liv Area' and target 'SalePrice'
X = df[['Gr Liv Area']].values
y = df['SalePrice'].values

from sklearn.preprocessing import StandardScaler  # For feature scaling

# Standardize the feature and target values
sc_x = StandardScaler()
sc_y = StandardScaler()

X_std = sc_x.fit_transform(X)
y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten()

# Initialize and train the linear regression model using gradient descent
lr = LinearRegressionGD(eta=0.1)
lr.fit(X_std, y_std)

# Plot the loss curve (MSE vs. epochs)
plt.plot(range(1, lr.n_iter+1), lr.losses_)
plt.ylabel('MSE')
plt.xlabel('Epoch')
plt.show()

def lin_regplot(X, y, model):
    # Plot the data points and the linear regression line
    plt.scatter(X, y, c='steelblue', edgecolor='white', s=70)
    plt.plot(X, model.predict(X), color='black', lw=2)

# Plot the linear regression fit on standardized data
lin_regplot(X_std, y_std, lr)
plt.xlabel('Living area above ground (standardized)')
plt.ylabel('Sale Price (standardized)')
plt.show()

# Predict the standardized sale price for a house with 2500 square feet of living area
feature_std = sc_x.transform(np.array([[2500]]))
target_std = lr.predict(feature_std)
# Inverse transform to get the prediction in the original scale
target_reverted = sc_y.inverse_transform(target_std.reshape(-1, 1))
print(f'Sales price: ${target_reverted.flatten()[0]:.2f}')

# Print the slope (weight) and intercept (bias) of the gradient descent model
print(f'Slope: {lr.w_[0]:.3f}')
print(f'Intercept: {lr.b_[0]:.3f}')

# Estimating the coefficient of a regression model via scikit-learn
from sklearn.linear_model import LinearRegression

# Initialize and train scikit-learn's LinearRegression model on the raw data
slr = LinearRegression()
slr.fit(X, y)
y_pred = slr.predict(X)

# Print the estimated slope and intercept
print(f'Slope: {slr.coef_[0]:.3f}')
print(f'Intercept: {slr.intercept_:.3f}')

# Plot the regression line using scikit-learn's model
lin_regplot(X, y, slr)
plt.xlabel('Living area above ground in square feet')
plt.ylabel('Sale Price in US dollars')
plt.tight_layout()
plt.show()


################################################################################
# %% Fitting a robust regression model using RANSAC
# This section applies the RANSAC algorithm to fit a robust linear regression model,
# which is less sensitive to outliers. It visualizes inliers and outliers, and prints
# the model's parameters.
################################################################################

from sklearn.linear_model import RANSACRegressor

# Initialize the RANSAC regressor with a base estimator of LinearRegression and set parameters
ransac = RANSACRegressor(
    LinearRegression(),
    max_trials=100,
    min_samples=0.95,
    residual_threshold=65000,
    random_state=123
)

# Fit the RANSAC regressor on the original data
ransac.fit(X, y)

# Obtain boolean masks indicating inlier and outlier samples
inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)

# Define a range for plotting the regression line
line_X = np.arange(3, 10, 1)
line_y_ransac = ransac.predict(line_X[:, np.newaxis])

# Plot inliers, outliers, and the RANSAC regression line
plt.scatter(X[inlier_mask], y[inlier_mask], c='steelblue', edgecolor='white', marker='o', label='Inliers')
plt.scatter(X[outlier_mask], y[outlier_mask], c='limegreen', edgecolor='white', marker='s', label='Outliers')
plt.plot(line_X, line_y_ransac, color='black', lw=2)
plt.xlabel('Living area above ground in square feet')
plt.ylabel('Sale Price in US dollars')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

# Print the slope and intercept of the robust regression model obtained by RANSAC
print(f'Slope: {ransac.estimator_.coef_[0]:.3f}')
print(f'Intercept: {ransac.estimator_.intercept_:.3f}')

# Define a function to compute the median absolute deviation (MAD)
def median_absolute_deviation(data):
    return np.median(np.abs(data - np.median(data)))
print(median_absolute_deviation(y))


################################################################################
# %% Evaluating the performance of linear regression models
# This section splits the dataset into training and test sets using all features,
# fits a linear regression model, and evaluates its performance using residual plots,
# mean squared error (MSE), mean absolute error (MAE), and R² score.
################################################################################

from sklearn.model_selection import train_test_split

# Define the target variable and feature matrix (all columns except SalePrice)
target = 'SalePrice'
features = df.columns[df.columns != target]
X = df[features].values
y = df[target].values

# Split data into training and test sets (70% training, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

# Fit a linear regression model on the training data
slr = LinearRegression()
slr.fit(X_train, y_train)
y_train_pred = slr.predict(X_train)
y_test_pred = slr.predict(X_test)

# Determine the minimum and maximum predicted values for plotting reference lines
x_max = np.max([np.max(y_train_pred), np.max(y_test_pred)])
x_min = np.min([np.min(y_train_pred), np.min(y_test_pred)])

# Create subplots for residual plots of test and training data
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3), sharey=True)
ax1.scatter(
    y_test_pred, y_test_pred - y_test,
    c='limegreen', marker='s',
    edgecolor='white',
    label='Test Data'
)
ax2.scatter(
    y_train_pred, y_train_pred - y_train,
    c='steelblue', marker='o',
    edgecolor='white',
    label='Training Data'
)
ax1.set_ylabel('Residuals')

# Set labels and reference lines on both subplots
for ax in (ax1, ax2):
    ax.set_xlabel('Predicted values')
    ax.legend(loc='upper left')
    ax.hlines(y=0, xmin=x_min - 100, xmax=x_max + 100, color='black', lw=2)

plt.tight_layout()
plt.show()

# Compute and print performance metrics: MSE, MAE, and R² score for training and test sets
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
print(f'MSE train: {mse_train:.2f}')
print(f'MSE test: {mse_test:.2f}')

mae_train = mean_absolute_error(y_train, y_train_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)
print(f'MAE train: {mae_train:.2f}')
print(f'MAE test: {mae_test:.2f}')

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
print(f'R2 train: {train_r2:.2f}')
print(f'R2 test: {test_r2:.2f}')


################################################################################
# %% Using regularized methods for regression
# This section imports regularized regression methods: Ridge, Lasso, and ElasticNet.
# (Implementation details are omitted, but these models help prevent overfitting.)
################################################################################

from sklearn.linear_model import Ridge
ridge = Ridge(alpha=1.0)

from sklearn.linear_model import Lasso
lasso = Lasso(alpha=1.0)

from sklearn.linear_model import ElasticNet
elanet = ElasticNet(alpha=1.0, l1_ratio=0.5)


################################################################################
# %% Turning a linear regression model into a curve - polynomial regression
# This section demonstrates how to fit polynomial regression models by transforming
# the input features using PolynomialFeatures. Both linear and quadratic (or cubic)
# fits are compared using MSE and R² scores.
################################################################################

# Example with a small synthetic dataset
from sklearn.preprocessing import PolynomialFeatures

# Define sample data points for X and y
X = np.array([259.0, 270.0, 294.0, 320.0, 342.0,
              368.0, 396.0, 446.0, 480.0, 586.0])[:, np.newaxis]
y = np.array([236.4, 234.4, 252.8, 298.6, 314.2,
              342.2, 360.8, 368.0, 391.1, 390.8])

# Initialize LinearRegression models for linear and polynomial fits
lr = LinearRegression()
pr = LinearRegression()

# Create polynomial features of degree 2 (quadratic)
quadratic = PolynomialFeatures(degree=2)
X_quad = quadratic.fit_transform(X)

# Fit the linear regression model on original data
lr.fit(X, y)
# Generate a range of values for plotting the fit
X_fit = np.arange(250, 600, 10)[:, np.newaxis]
y_lin_fit = lr.predict(X_fit)

# Fit the polynomial regression model on quadratic features
pr.fit(X_quad, y)
y_quad_fit = pr.predict(quadratic.fit_transform(X_fit))

# Plot the original training points, linear fit, and quadratic fit
plt.scatter(X, y, label='Training points')
plt.plot(X_fit, y_lin_fit, label='Linear fit', linestyle='--')
plt.plot(X_fit, y_quad_fit, label='Quadratic fit')
plt.xlabel('Explanatory variable')
plt.ylabel('Predicted or known target values')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

# Calculate and print training MSE and R² for linear vs. quadratic models
y_lin_pred = lr.predict(X)
y_quad_pred = pr.predict(X_quad)
from sklearn.metrics import mean_squared_error
mse_lin = mean_squared_error(y, y_lin_pred)
mse_quad = mean_squared_error(y, y_quad_pred)
print(f'Training MSE linear: {mse_lin:.3f}, quadratic: {mse_quad:.3f}')

r2_lin = r2_score(y, y_lin_pred)
r2_quad = r2_score(y, y_quad_pred)
print(f'Training R2 linear: {r2_lin:.3f}, quadratic: {r2_quad:.3f}')


# Modeling nonlinear relationships in the Ames Housing dataset
# Use 'Gr Liv Area' and 'Overall Qual' as explanatory variables separately
# First, for 'Gr Liv Area' (restricting to values < 4000)
X = df[['Gr Liv Area']].values
y = df['SalePrice'].values
X = X[(df['Gr Liv Area'] < 4000)]
y = y[(df['Gr Liv Area'] < 4000)]

regr = LinearRegression()
# Create quadratic and cubic features
quadratic = PolynomialFeatures(degree=2)
cubic = PolynomialFeatures(degree=3)
X_quad = quadratic.fit_transform(X)
X_cubic = cubic.fit_transform(X)
# Generate a range for plotting the fitted curves
X_fit = np.arange(X.min()-1, X.max()+2, 1)[:, np.newaxis]
# Linear fit
regr = regr.fit(X, y)
y_lin_fit = regr.predict(X_fit)
linear_r2 = r2_score(y, regr.predict(X))
# Quadratic fit
regr = regr.fit(X_quad, y)
y_quad_fit = regr.predict(quadratic.fit_transform(X_fit))
quadratic_r2 = r2_score(y, regr.predict(X_quad))
# Cubic fit
regr = regr.fit(X_cubic, y)
y_cubic_fit = regr.predict(cubic.fit_transform(X_fit))
cubic_r2 = r2_score(y, regr.predict(X_cubic))
# Plot all fits for comparison
plt.scatter(X, y, label='Training points', color='lightgray')
plt.plot(X_fit, y_lin_fit, label=f'Linear (d=1), $R^2$={linear_r2:.2f}', color='blue', lw=2, linestyle=':')
plt.plot(X_fit, y_quad_fit, label=f'Quadratic (d=2), $R^2$={quadratic_r2:.2f}', color='red', lw=2, linestyle='-')
plt.plot(X_fit, y_cubic_fit, label=f'Cubic (d=3), $R^2$={cubic_r2:.2f}', color='green', lw=2, linestyle='--')
plt.xlabel('Living area above ground in square feet')
plt.ylabel('Sale price in U.S. dollars')
plt.legend(loc='upper left')
plt.show()

# Next, for 'Overall Qual'
X = df[['Overall Qual']].values
y = df['SalePrice'].values

regr = LinearRegression()
# Create quadratic and cubic features for 'Overall Qual'
quadratic = PolynomialFeatures(degree=2)
cubic = PolynomialFeatures(degree=3)
X_quad = quadratic.fit_transform(X)
X_cubic = cubic.fit_transform(X)
# Generate a range for plotting the fit
X_fit = np.arange(X.min()-1, X.max()+2, 1)[:, np.newaxis]
# Linear fit
regr = regr.fit(X, y)
y_lin_fit = regr.predict(X_fit)
linear_r2 = r2_score(y, regr.predict(X))
# Quadratic fit
regr = regr.fit(X_quad, y)
y_quad_fit = regr.predict(quadratic.fit_transform(X_fit))
quadratic_r2 = r2_score(y, regr.predict(X_quad))
# Cubic fit
regr = regr.fit(X_cubic, y)
y_cubic_fit = regr.predict(cubic.fit_transform(X_fit))
cubic_r2 = r2_score(y, regr.predict(X_cubic))
# Plot all fits
plt.scatter(X, y, label='Training points', color='lightgray')
plt.plot(X_fit, y_lin_fit, label=f'Linear (d=1), $R^2$={linear_r2:.2f}', color='blue', lw=2, linestyle=':')
plt.plot(X_fit, y_quad_fit, label=f'Quadratic (d=2), $R^2$={quadratic_r2:.2f}', color='red', lw=2, linestyle='-')
plt.plot(X_fit, y_cubic_fit, label=f'Cubic (d=3), $R^2$={cubic_r2:.2f}', color='green', lw=2, linestyle='--')
plt.xlabel('Overall Qual')
plt.ylabel('Sale price in U.S. dollars')
plt.legend(loc='upper left')
plt.show()


################################################################################
# %% Dealing with nonlinear relationships using random forests
# This section first demonstrates nonlinear regression using a Decision Tree Regressor
# and then a Random Forest Regressor on the 'Gr Liv Area' feature to predict sale prices.
################################################################################

from sklearn.tree import DecisionTreeRegressor

# Use 'Gr Liv Area' as the feature and 'SalePrice' as the target
X = df[['Gr Liv Area']].values
y = df['SalePrice'].values

# Initialize and fit a Decision Tree Regressor with max depth of 3
tree = DecisionTreeRegressor(max_depth=3)
tree.fit(X, y)

# Sort the X values for plotting the regression curve smoothly
sort_idx = X.flatten().argsort()

# Plot the decision tree regression results using the custom lin_regplot function defined earlier
lin_regplot(X[sort_idx], y[sort_idx], tree)
plt.xlabel('Living area above ground in square feet')
plt.ylabel('Sale price in U.S. dollars')
plt.show()

# Random Forest regression using multiple features
target = 'SalePrice'
features = df.columns[df.columns != target]

X = df[features].values
y = df[target].values

# Split the dataset into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

from sklearn.ensemble import RandomForestRegressor

# Initialize the Random Forest Regressor with 1000 trees
forest = RandomForestRegressor(
    n_estimators=1000,
    criterion='squared_error',
    random_state=1,
    n_jobs=-1
)

# Fit the model on the training data
forest.fit(X_train, y_train)
# Predict on both training and test sets
y_train_pred = forest.predict(X_train)
y_test_pred = forest.predict(X_test)

# Compute Mean Absolute Error for training and test sets
from sklearn.metrics import mean_absolute_error
mae_train = mean_absolute_error(y_train, y_train_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)
print(f'MAE train: {mae_train:.2f}')
print(f'MAE test: {mae_test:.2f}')

# Compute R2 score for training and test sets
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)
print(f'R2 train: {r2_train:.2f}')
print(f'R2 test: {r2_test:.2f}')

# Determine the plotting range based on predictions
x_max = np.max([np.max(y_train_pred), np.max(y_test_pred)])
x_min = np.min([np.min(y_train), np.min(y_test_pred)])

# Create residual plots for training and test data side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3), sharey=True)
ax1.scatter(y_test_pred, y_test_pred - y_test, c='limegreen', marker='s', edgecolor='white', label='Test data')
ax2.scatter(y_train_pred, y_train_pred - y_train, c='steelblue', marker='o', edgecolor='white', label='Training data')
ax1.set_ylabel('Residuals')

# Add horizontal line at zero residual for reference on both plots
for ax in (ax1, ax2):
    ax.set_xlabel('Predicted values')
    ax.legend(loc='upper left')
    ax.hlines(y=0, xmin=x_min - 100, xmax=x_max + 100, color='black', lw=2)

plt.tight_layout()
plt.show()
