import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

np.random.seed(42)

def k_means(X, k, max_iterations):
    """
    K-Means Clustering Algorithm
    Parameters:
    - X: Input data
    - k: Number of clusters
    - max_iterations: Max number of iterations
    Returns:
    - clusters: Clustered data (array of numpy arrays)
    - centroids: Centroids of clusters (numpy array of arrays)
    """
    X = np.array(X)
    n_samples, size = X.shape
    
    # Initialize centroids by randomly selecting k points from X
    indices = np.random.choice(n_samples, k, replace=False)
    centroids = X[indices]
    
    print(f'Initial random centroids:\n{centroids}')
    
    for j in range(max_iterations):
        # Initialize clusters for this iteration
        clusters = [[] for _ in range(k)]
        
        # Assign each point to the nearest centroid
        for x in X:
            distances = [np.sqrt(np.sum((x - centroid) ** 2)) for centroid in centroids]
            cluster_idx = np.argmin(distances)
            clusters[cluster_idx].append(x)
        
        # Store old centroids for convergence check
        old_centroids = centroids.copy()
        
        # Update centroids based on mean of each cluster
        new_centroids = []
        for cluster in clusters:
            if len(cluster) > 0:  # Handle non-empty clusters
                new_centroid = np.mean(np.array(cluster), axis=0)
            else:  # Keep old centroid for empty clusters
                new_centroid = old_centroids[len(new_centroids)]
            new_centroids.append(new_centroid)
        
        centroids = np.array(new_centroids)
        print(f'Iteration {j + 1}, new centroids:\n{centroids}')
        
        # Check for convergence
        if np.allclose(old_centroids, centroids):
            print("Converged!")
            break
    
    clusters = [np.array(cluster) for cluster in clusters]
    return clusters, centroids