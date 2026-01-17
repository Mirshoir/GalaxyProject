#clustering.py
"""
Clustering in distance space.
Uses k-medoids algorithm which works directly with distance matrices.
"""
import numpy as np
from typing import Tuple, List, Optional
from sklearn.metrics import pairwise_distances
from sklearn_extra.cluster import KMedoids
import warnings

class DistanceSpaceClustering:
    """Clustering based on precomputed distance matrix."""

    def __init__(self,
                 n_clusters: int = 3,
                 random_state: int = 42,
                 method: str = 'pam'):
        """
        Args:
            n_clusters: Number of clusters
            random_state: Random seed for reproducibility
            method: 'pam' (Partitioning Around Medoids) or 'alternate'
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.method = method
        self.model = None
        self.labels_ = None
        self.medoids_ = None

    def fit(self, distance_matrix: np.ndarray) -> 'DistanceSpaceClustering':
        """
        Fit clustering model to distance matrix.

        Args:
            distance_matrix: Precomputed N x N distance matrix

        Returns:
            self
        """
        if distance_matrix.shape[0] != distance_matrix.shape[1]:
            raise ValueError("Distance matrix must be square")

        # Initialize and fit k-medoids
        self.model = KMedoids(
            n_clusters=self.n_clusters,
            metric='precomputed',
            method=self.method,
            init='k-medoids++',
            random_state=self.random_state
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(distance_matrix)

        self.labels_ = self.model.labels_
        self.medoids_ = self.model.medoid_indices_

        return self

    def predict(self, distance_matrix: np.ndarray) -> np.ndarray:
        """Predict cluster labels for new distances."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # For k-medoids with precomputed distances, we can compute
        # distances to medoids and assign to closest
        n_samples = distance_matrix.shape[0]
        n_clusters = len(self.medoids_)

        # Create distance matrix from each sample to each medoid
        medoid_distances = np.zeros((n_samples, n_clusters))

        for i in range(n_samples):
            for j, medoid_idx in enumerate(self.medoids_):
                if i < distance_matrix.shape[1]:
                    medoid_distances[i, j] = distance_matrix[i, medoid_idx]
                else:
                    # For out-of-sample, we need to compute distances
                    # This is a limitation of k-medoids
                    raise ValueError("Out-of-sample prediction not supported "
                                   "with precomputed distances")

        return np.argmin(medoid_distances, axis=1)

    def get_cluster_members(self, cluster_id: int) -> List[int]:
        """Get indices of samples in a specific cluster."""
        if self.labels_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        return np.where(self.labels_ == cluster_id)[0].tolist()

    def get_cluster_sizes(self) -> np.ndarray:
        """Get size of each cluster."""
        if self.labels_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        unique, counts = np.unique(self.labels_, return_counts=True)
        return counts

def hierarchical_clustering(distance_matrix: np.ndarray,
                           n_clusters: int = 3,
                           linkage: str = 'average') -> np.ndarray:
    """
    Alternative: Hierarchical clustering using distance matrix.

    Args:
        distance_matrix: Precomputed N x N distance matrix
        n_clusters: Number of clusters
        linkage: Linkage method ('average', 'complete', 'single', 'ward')

    Returns:
        Cluster labels
    """
    from scipy.cluster.hierarchy import linkage as scipy_linkage, fcluster

    # Convert condensed distance matrix
    condensed_dist = distance_matrix[np.triu_indices(distance_matrix.shape[0], k=1)]

    # Perform hierarchical clustering
    Z = scipy_linkage(condensed_dist, method=linkage)

    # Form flat clusters
    labels = fcluster(Z, n_clusters, criterion='maxclust')

    # Convert to 0-indexed
    return labels - 1

def create_distance_histogram(distance_matrix: np.ndarray,
                            cluster_labels: np.ndarray,
                            cluster_id: int) -> np.ndarray:
    """
    Create histogram of distances within a cluster.

    Useful for analyzing cluster compactness.
    """
    cluster_indices = np.where(cluster_labels == cluster_id)[0]

    if len(cluster_indices) < 2:
        return np.array([])

    # Get intra-cluster distances
    intra_distances = []
    for i in range(len(cluster_indices)):
        for j in range(i+1, len(cluster_indices)):
            idx1, idx2 = cluster_indices[i], cluster_indices[j]
            intra_distances.append(distance_matrix[idx1, idx2])

    return np.array(intra_distances)