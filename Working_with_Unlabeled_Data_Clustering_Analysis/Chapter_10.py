"""
Chapter 10: Working with Unlabeled Data - Clustering Analysis
Luke Bray
February 13, 2025
"""

################################################################################
# %% Grouping objects by similarity using k-means
# In this section, we generate synthetic data using make_blobs and perform k-means
# clustering on the data. We visualize the raw data, the clustering results, and
# also use the elbow method to determine the optimal number of clusters. Additionally,
# we compute silhouette scores to evaluate cluster quality.
################################################################################

# Import necessary libraries for generating synthetic data and plotting
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np

# Generate a synthetic dataset with 150 samples, 2 features, and 3 centers
X, y = make_blobs(
    n_samples=150,
    n_features=2,
    centers=3,
    cluster_std=0.5,
    shuffle=True,
    random_state=40
)

# Plot the generated data points as white circles with black edges
plt.scatter(
    X[:, 0],
    X[:, 1],
    c='white',
    marker='o',
    edgecolor='black',
    s=50
)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid()
plt.tight_layout()
plt.show()

# Import KMeans from scikit-learn
from sklearn.cluster import KMeans

# Initialize KMeans with 3 clusters, using random initialization parameters
km = KMeans(
    n_clusters=3,
    init='random',
    n_init=10,
    max_iter=300,
    tol=1e-04,
    random_state=0
)

# Fit KMeans on the dataset and predict cluster indices
y_km = km.fit_predict(X)

# Plot the clustering results: each cluster in a different color and marker
plt.scatter(
    X[y_km == 0, 0],
    X[y_km == 0, 1],
    s=50,
    c='lightgreen',
    marker='s',
    edgecolor='black',
    label='Cluster 1'
)
plt.scatter(
    X[y_km == 1, 0],
    X[y_km == 1, 1],
    s=50,
    c='orange',
    marker='o',
    edgecolor='black',
    label='Cluster 2'
)
plt.scatter(
    X[y_km == 2, 0],
    X[y_km == 2, 1],
    s=50,
    c='lightblue',
    marker='v',
    edgecolor='black',
    label='Cluster 3'
)

# Plot the centroids of the clusters as red stars
plt.scatter(
    km.cluster_centers_[:, 0],
    km.cluster_centers_[:, 1],
    s=250,
    marker='*',
    c='red',
    edgecolor='black',
    label='Centroids'
)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend(scatterpoints=1)
plt.grid()
plt.tight_layout()
plt.show()

# Demonstration of a smarter initialization method using k-means++
# (Commented out here; to use, change init from 'random' to 'k-means++')
"""
km = KMeans(
    n_clusters=3,
    init='k-means++',
    n_init=10,
    max_iter=300,
    tol=1e-04,
    random_state=0
)
"""

# Use the elbow method to determine the optimal number of clusters
print(f'Distortion: {km.inertia_:.2f}')

# Calculate distortion (sum of squared distances) for cluster counts 1 to 10
distortions = []
for i in range(1, 11):
    km = KMeans(
        n_clusters=i,
        init='k-means++',
        n_init=10,
        max_iter=300,
        random_state=0
    )
    km.fit(X)
    distortions.append(km.inertia_)

# Plot distortion values vs. number of clusters to find the "elbow"
plt.plot(range(1, 11), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.grid()
plt.tight_layout()
plt.show()

# Quantifying clustering quality using silhouette plots
# Re-run KMeans with k-means++ initialization for 3 clusters
km = KMeans(
    n_clusters=3,
    init='k-means++',
    n_init=10,
    max_iter=300,
    tol=1e-04,
    random_state=0
)
y_km = km.fit_predict(X)

from matplotlib import cm
from sklearn.metrics import silhouette_samples

# Calculate silhouette scores for each sample using Euclidean distance
cluster_labels = np.unique(y_km)
n_clusters = cluster_labels.shape[0]
silhouette_vals = silhouette_samples(X, y_km, metric='euclidean')

# Prepare variables for plotting silhouette bars
y_ax_lower, y_ax_upper = 0, 0
yticks = []

# Plot silhouette values for each cluster
for i, c in enumerate(cluster_labels):
    c_silhouette_vals = silhouette_vals[y_km == c]
    c_silhouette_vals.sort()
    y_ax_upper += len(c_silhouette_vals)
    color = cm.jet(float(i) / n_clusters)
    plt.barh(
        range(y_ax_lower, y_ax_upper),
        c_silhouette_vals,
        height=1.0,
        edgecolor='none',
        color=color
    )
    yticks.append((y_ax_lower + y_ax_upper) / 2)
    y_ax_lower += len(c_silhouette_vals)

# Compute average silhouette score and plot it as a vertical line
silhouette_avg = np.mean(silhouette_vals)
plt.axvline(
    silhouette_avg,
    color='red',
    linestyle='--'
)
plt.yticks(yticks, cluster_labels + 1)
plt.ylabel('Cluster')
plt.xlabel('Silhouette Coefficient')
plt.tight_layout()
plt.show()

# Repeat clustering and silhouette analysis for k=2 clusters
km = KMeans(n_clusters=2,
            init='k-means++',
            n_init=10,
            max_iter=300,
            tol=1e-04,
            random_state=0)
y_km = km.fit_predict(X)
plt.scatter(X[y_km == 0, 0],
            X[y_km == 0, 1],
            s=50, c='lightgreen',
            edgecolor='black',
            marker='s',
            label='Cluster 1')
plt.scatter(X[y_km == 1, 0],
            X[y_km == 1, 1],
            s=50,
            c='orange',
            edgecolor='black',
            marker='o',
            label='Cluster 2')
plt.scatter(km.cluster_centers_[:, 0],
            km.cluster_centers_[:, 1],
            s=250,
            marker='*',
            c='red',
            label='Centroids')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# Compute silhouette values for k=2 clusters and plot
cluster_labels = np.unique(y_km)
n_clusters = cluster_labels.shape[0]
silhouette_vals = silhouette_samples(X, y_km, metric='euclidean')
y_ax_lower, y_ax_upper = 0, 0
yticks = []
for i, c in enumerate(cluster_labels):
    c_silhouette_vals = silhouette_vals[y_km == c]
    c_silhouette_vals.sort()
    y_ax_upper += len(c_silhouette_vals)
    color = cm.jet(float(i) / n_clusters)
    plt.barh(range(y_ax_lower, y_ax_upper),
             c_silhouette_vals,
             height=1.0,
             edgecolor='none',
             color=color)
    yticks.append((y_ax_lower + y_ax_upper) / 2.)
    y_ax_lower += len(c_silhouette_vals)
silhouette_avg = np.mean(silhouette_vals)
plt.axvline(silhouette_avg, color="red", linestyle="--")
plt.yticks(yticks, cluster_labels + 1)
plt.ylabel('Cluster')
plt.xlabel('Silhouette coefficient')
plt.tight_layout()
plt.show()


################################################################################
# %% Organizing clusters as a hierarchical tree
# This section performs hierarchical clustering on a small, randomly generated dataset.
# It computes a distance matrix, performs hierarchical (agglomerative) clustering,
# and visualizes the clustering results using dendrograms. Finally, it attaches the
# dendrogram to a heat map for enhanced visualization.
################################################################################

import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(123)
# Define variables and labels for the dataset
variables = ['X', 'Y', 'Z']
labels = ['ID_0', 'ID_1', 'ID_2', 'ID_3', 'ID_4']

# Generate a 5x3 matrix of random numbers scaled by 10
X = np.random.random_sample([5, 3]) * 10

# Create a DataFrame with the generated data
df = pd.DataFrame(X, columns=variables, index=labels)
print(df)

# Compute the pairwise Euclidean distances between rows and convert to a square matrix
from scipy.spatial.distance import pdist, squareform
row_dist = pd.DataFrame(squareform(pdist(df, metric='euclidean')), columns=labels, index=labels)
print(row_dist)

# Perform hierarchical clustering using the complete linkage method
from scipy.cluster.hierarchy import linkage
row_clusters = linkage(df.values,
                       method='complete',
                       metric='euclidean')

# Create a DataFrame to display the linkage matrix with descriptive column names
pd.DataFrame(
    row_clusters,
    columns=[
        'row label 1',
        'row label 2',
        'distance',
        'no. of items in cluster'
        ],
    index=[f'cluster {(i + 1)}' for i in range(row_clusters.shape[0])]
)

# Plot a dendrogram to visualize the hierarchical clustering structure
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import set_link_color_palette

# Set dendrogram link colors to black
set_link_color_palette(['black'])
row_dendr = dendrogram(
    row_clusters,
    labels=labels
)
plt.tight_layout()
plt.show()

# Attach the dendrogram to a heat map
fig = plt.figure(figsize=(8, 8), facecolor='white')
axd = fig.add_axes([0.09, 0.1, 0.2, 0.6])
row_dendr = dendrogram(row_clusters,
                       orientation='left')
# Reorder the DataFrame based on the dendrogram leaves
df_rowclust = df.iloc[row_dendr['leaves'][::-1]]

axm = fig.add_axes([0.23, 0.1, 0.6, 0.6])
# Display the reordered DataFrame as a heat map
cax = axm.matshow(
    df_rowclust,
    interpolation='nearest',
    cmap='hot_r'
)
# Remove ticks and spines from the dendrogram plot
axd.set_xticks([])
axd.set_yticks([])
for i in axd.spines.values():
    i.set_visible(False)
fig.colorbar(cax)
axm.set_xticklabels([''] + list(df_rowclust.columns))
axm.set_yticklabels([''] + list(df_rowclust.index))
plt.show()

# Apply agglomerative clustering using scikit-learn
from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(
    n_clusters=3,
    linkage='complete'
)
labels_ac = ac.fit_predict(X)
print(f'Cluster labels: {labels_ac}')

# Run agglomerative clustering with 2 clusters and print results
ac = AgglomerativeClustering(
    n_clusters=2,
    linkage='complete'
)
labels_ac = ac.fit_predict(X)
print(f'Cluster labels: {labels_ac}')


################################################################################
# %% Locating regions of high density via DBSCAN
# This section applies DBSCAN to the "moons" dataset to locate regions of high
# density. First, it creates a two-dimensional dataset using make_moons,
# then compares clustering results using K-means, Agglomerative Clustering, and DBSCAN.
################################################################################

from sklearn.datasets import make_moons

# Generate a two-dimensional dataset with two moon-shaped clusters and noise
X, y = make_moons(
    n_samples=200,
    noise=0.05,
    random_state=0
)

# Plot the raw moons data
plt.scatter(X[:, 0], X[:, 1])
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.tight_layout()
plt.show()

# Create subplots to compare K-means and Agglomerative clustering on the moons data
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
# K-means clustering with 2 clusters
km = KMeans(n_clusters=2, random_state=0)
y_km = km.fit_predict(X)
ax1.scatter(X[y_km == 0, 0],
            X[y_km == 0, 1],
            c='lightblue',
            edgecolor='black',
            marker='o',
            s=40,
            label='Cluster 1')
ax1.scatter(X[y_km == 1, 0],
            X[y_km == 1, 1],
            c='red',
            edgecolor='black',
            marker='s',
            s=40,
            label='Cluster 2')
ax1.set_title('K-means clustering')
ax1.set_xlabel('Feature 1')
ax1.set_ylabel('Feature 2')

# Agglomerative clustering with 2 clusters
ac = AgglomerativeClustering(n_clusters=2, linkage='complete')
y_ac = ac.fit_predict(X)
ax2.scatter(X[y_ac == 0, 0],
            X[y_ac == 0, 1],
            c='lightblue',
            edgecolor='black',
            marker='o',
            s=40,
            label='Cluster 1')
ax2.scatter(X[y_ac == 1, 0],
            X[y_ac == 1, 1],
            c='red',
            edgecolor='black',
            marker='s',
            s=40,
            label='Cluster 2')
ax2.set_title('Agglomerative clustering')
ax2.set_xlabel('Feature 1')
ax2.set_ylabel('Feature 2')
plt.legend()
plt.tight_layout()
plt.show()

# Apply DBSCAN to detect clusters based on density
from sklearn.cluster import DBSCAN
db = DBSCAN(eps=0.2,
            min_samples=5,
            metric='euclidean')
y_db = db.fit_predict(X)
# Plot DBSCAN clustering results with different colors for clusters
plt.scatter(X[y_db == 0, 0],
            X[y_db == 0, 1],
            c='lightblue',
            edgecolor='black',
            marker='o',
            s=40,
            label='Cluster 1')
plt.scatter(X[y_db == 1, 0],
            X[y_db == 1, 1],
            c='red',
            edgecolor='black',
            marker='s',
            s=40,
            label='Cluster 2')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.tight_layout()
plt.show()


################################################################################
# %% Organizing clusters as a hierarchical tree (dendrogram and heat map)
# This section demonstrates hierarchical clustering by creating a small dataset,
# computing a distance matrix, performing hierarchical clustering using complete linkage,
# and visualizing the results with a dendrogram. It also attaches the dendrogram to a
# heat map for a more comprehensive visualization.
################################################################################

import pandas as pd
import numpy as np

# Set a random seed for reproducibility
np.random.seed(123)
# Define variable names and labels for a small dataset
variables = ['X', 'Y', 'Z']
labels = ['ID_0', 'ID_1', 'ID_2', 'ID_3', 'ID_4']

# Generate random data and scale it by 10
X = np.random.random_sample([5, 3]) * 10

# Create a DataFrame with the generated data
df = pd.DataFrame(X, columns=variables, index=labels)
print(df)

# Compute the pairwise Euclidean distance matrix between rows and convert it to a DataFrame
from scipy.spatial.distance import pdist, squareform
row_dist = pd.DataFrame(squareform(pdist(df, metric='euclidean')), columns=labels, index=labels)
print(row_dist)

# Perform hierarchical clustering using the complete linkage method
from scipy.cluster.hierarchy import linkage
row_clusters = linkage(df.values,
                       method='complete',
                       metric='euclidean')

# Create a DataFrame of the clustering results with descriptive column names
pd.DataFrame(
    row_clusters,
    columns=[
        'row label 1',
        'row label 2',
        'distance',
        'no. of items in cluster'
        ],
    index=[f'cluster {(i + 1)}' for i in range(row_clusters.shape[0])]
)

# Plot a dendrogram to visualize the hierarchical clustering
from scipy.cluster.hierarchy import dendrogram, set_link_color_palette
set_link_color_palette(['black'])
row_dendr = dendrogram(
    row_clusters,
    labels=labels
)
plt.tight_layout()
plt.show()

# Attach the dendrogram to a heat map for enhanced visualization
fig = plt.figure(figsize=(8, 8), facecolor='white')
axd = fig.add_axes([0.09, 0.1, 0.2, 0.6])
row_dendr = dendrogram(row_clusters, orientation='left')
df_rowclust = df.iloc[row_dendr['leaves'][::-1]]
axm = fig.add_axes([0.23, 0.1, 0.6, 0.6])
cax = axm.matshow(df_rowclust, interpolation='nearest', cmap='hot_r')
axd.set_xticks([])
axd.set_yticks([])
for spine in axd.spines.values():
    spine.set_visible(False)
fig.colorbar(cax)
axm.set_xticklabels([''] + list(df_rowclust.columns))
axm.set_yticklabels([''] + list(df_rowclust.index))
plt.show()

# Apply agglomerative clustering using scikit-learn on the small dataset
from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(
    n_clusters=3,
    linkage='complete'
)
labels_ac = ac.fit_predict(X)
print(f'Cluster labels: {labels_ac}')

ac = AgglomerativeClustering(
    n_clusters=2,
    linkage='complete'
)
labels_ac = ac.fit_predict(X)
print(f'Cluster labels: {labels_ac}')