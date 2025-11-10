import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score

def create_data(n_samples, n_clusters):
    """
    Creates synthetic data for clustering demonstration.

    Returns:
        np.ndarray: The generated data points.
        np.ndarray: The corresponding cluster labels.
    """
    from sklearn.datasets import make_blobs
    X, Y = make_blobs(n_samples=n_samples, centers=n_clusters, cluster_std=0.6, center_box=(-5,5), random_state=420)
    return X, Y

# --- Main Script ---
n_samples = 2000  # Number of data points
n_clusters = 5  # Number of clusters
k = 4  # Number of clusters for k-means
num_initialisations = 4  # Number of random initialisations to try

# Generate synthetic data
X, Y_true = create_data(n_samples, n_clusters)

# Set up plotting
fig = plt.figure(dpi=200,figsize=(7, 7))
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)
axs = [ax1, ax2, ax3, ax4]

# Apply K-Means clustering
# Take starting centroids from random data points (Forgy initialisation)
# Centroids shape: (num_initialisations, 2, k)
centroids = np.array([X[np.random.choice(X.shape[0], k, replace=False) , :].T for _ in range(num_initialisations)])
converged = False
iter = 0
while not converged:
    # Find distances from points to centroids
    # We use broadcasting to compute the distance matrix efficiently
    distances = np.linalg.norm(X[None, :, :, np.newaxis] - centroids[:, None, :, :], axis=2)
    # distances is shape (num_initialisations, n_samples, k)
    # Assign clusters based on closest centroid
    Y_pred = np.argmin(distances, axis=2)

    # Update centroids
    new_centroids = []
    for n in range(num_initialisations):
        new_centroid_n = []
        for i in range(k):
            # Centroid of cluster i is the mean of all points assigned to it
            new_centroid_i = np.mean(X[Y_pred[n,:] == i], axis=0)
            new_centroid_n.append(new_centroid_i)
        new_centroids.append(np.array(new_centroid_n).T)  # Shape (2, k)
    new_centroids = np.array(new_centroids)  # Shape (num_initialisations, 2, k)

    for n,ax in enumerate(axs):
        ax.clear()
        ax.scatter(X[:, 0], X[:, 1], c=Y_pred[n,:], cmap='tab10', alpha=0.5, s=10)
        ax.scatter(new_centroids[n, 0, :], new_centroids[n, 1, :], c='k', marker='X', s=20, label='Centroids')
        ax.set_title('Accuracy: {:.2f}%'.format(adjusted_rand_score(Y_true,Y_pred[n,:])*100.0))
        ax.legend()
    fig.savefig(f'./Images/KMeansClustering_{iter:03d}.png')

    # Check for convergence
    if np.isclose(centroids,new_centroids,rtol=1e-6,atol=1e-6).all():
        converged = True

    iter += 1
    centroids = new_centroids.copy()

