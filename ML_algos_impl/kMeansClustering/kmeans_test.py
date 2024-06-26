


### for a python implementation , you can checkout unsupervised_folder!! ###

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

# Generate some sample data
np.random.seed(0)
X = np.random.randn(100, 2)  # 100 samples, 2 features

# Create a KMeans instance with the desired number of clusters
kmeans = KMeans(n_clusters=3)

# Fit the KMeans model to the data
kmeans.fit(X)

# Get the cluster centroids and labels
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

# Plot the data points and cluster centroids
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='o', s=200, color='red', label='Centroids')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-means Clustering')
plt.legend()
plt.show()
