"""
Metrics for evaluating clustering quality.
Focuses on RMS distance within clusters.
"""
import numpy as np
from typing import List, Tuple, Dict
from scipy.spatial.distance import squareform
import warnings

def rms_within_cluster(distance_matrix: np.ndarray,
                      labels: np.ndarray,
                      cluster_id: int) -> float:
    """
    Compute RMS (root mean square) distance within a single cluster.

    Args:
        distance_matrix: Precomputed N x N distance matrix
        labels: Cluster assignment for each sample (0-indexed)
        cluster_id: ID of cluster to analyze

    Returns:
        RMS distance within the specified cluster
    """
    # Get indices of samples in this cluster
    cluster_indices = np.where(labels == cluster_id)[0]
    n_samples = len(cluster_indices)

    if n_samples < 2:
        return 0.0

    # Extract intra-cluster distances
    intra_distances = []
    for i in range(n_samples):
        for j in range(i+1, n_samples):
            idx1, idx2 = cluster_indices[i], cluster_indices[j]
            intra_distances.append(distance_matrix[idx1, idx2])

    if len(intra_distances) == 0:
        return 0.0

    # Compute RMS
    intra_distances = np.array(intra_distances)
    rms = np.sqrt(np.mean(intra_distances ** 2))

    return rms

def average_rms_across_clusters(distance_matrix: np.ndarray,
                               labels: np.ndarray,
                               weighted: bool = True) -> Tuple[float, Dict]:
    """
    Compute average RMS distance across all clusters.

    Args:
        distance_matrix: Precomputed N x N distance matrix
        labels: Cluster assignment for each sample
        weighted: If True, weight by cluster size

    Returns:
        Tuple of (average_rms, per_cluster_rms_dict)
    """
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    per_cluster_rms = {}
    cluster_sizes = {}

    # Compute RMS for each cluster
    for label in unique_labels:
        rms = rms_within_cluster(distance_matrix, labels, label)
        per_cluster_rms[label] = rms

        # Get cluster size
        size = np.sum(labels == label)
        cluster_sizes[label] = size

    # Compute weighted average
    if weighted:
        total_samples = len(labels)
        weighted_sum = 0

        for label in unique_labels:
            weight = cluster_sizes[label] / total_samples
            weighted_sum += per_cluster_rms[label] * weight

        average_rms = weighted_sum
    else:
        # Simple average
        average_rms = np.mean(list(per_cluster_rms.values()))

    return average_rms, per_cluster_rms

def compute_rms_vs_k(distance_matrix: np.ndarray,
                    k_values: List[int],
                    n_runs: int = 5,
                    random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute RMS vs k for elbow analysis.

    Args:
        distance_matrix: Precomputed N x N distance matrix
        k_values: List of k values to test
        n_runs: Number of random initializations per k
        random_state: Random seed

    Returns:
        Tuple of (k_values_array, mean_rms_array, std_rms_array)
    """
    from .clustering import DistanceSpaceClustering

    mean_rms_values = []
    std_rms_values = []

    for k in k_values:
        rms_values = []

        for run in range(n_runs):
            # Different random seed for each run
            current_seed = random_state + run

            # Cluster with k clusters
            clusterer = DistanceSpaceClustering(
                n_clusters=k,
                random_state=current_seed
            )
            clusterer.fit(distance_matrix)

            # Compute average RMS
            avg_rms, _ = average_rms_across_clusters(
                distance_matrix,
                clusterer.labels_,
                weighted=True
            )
            rms_values.append(avg_rms)

        mean_rms = np.mean(rms_values)
        std_rms = np.std(rms_values)

        mean_rms_values.append(mean_rms)
        std_rms_values.append(std_rms)

        print(f"k={k}: RMS = {mean_rms:.4f} ± {std_rms:.4f}")

    return np.array(k_values), np.array(mean_rms_values), np.array(std_rms_values)

def compute_silhouette_score(distance_matrix: np.ndarray,
                           labels: np.ndarray) -> float:
    """
    Compute silhouette score for clustering (higher is better).

    Note: This is computationally expensive for large datasets.
    """
    from sklearn.metrics import silhouette_score

    # Silhouette score requires pairwise distances
    # We'll sample if matrix is too large
    n_samples = distance_matrix.shape[0]

    if n_samples > 1000:
        # Sample for efficiency
        np.random.seed(42)
        sample_idx = np.random.choice(n_samples, 1000, replace=False)
        sample_dist = distance_matrix[np.ix_(sample_idx, sample_idx)]
        sample_labels = labels[sample_idx]
        return silhouette_score(sample_dist, sample_labels, metric='precomputed')
    else:
        return silhouette_score(distance_matrix, labels, metric='precomputed')

def compute_davies_bouldin_index(distance_matrix: np.ndarray,
                               labels: np.ndarray) -> float:
    """
    Compute Davies-Bouldin index (lower is better).

    Measures average similarity between each cluster and its most similar cluster.
    """
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    if n_clusters < 2:
        return 0.0

    # Compute cluster centroids in distance space (use medoids)
    from .clustering import DistanceSpaceClustering

    # Find medoid for each cluster
    cluster_medoids = []
    cluster_dispersions = []

    for label in unique_labels:
        cluster_indices = np.where(labels == label)[0]

        if len(cluster_indices) == 0:
            continue

        # Find medoid (point with minimum average distance to others in cluster)
        cluster_distances = distance_matrix[np.ix_(cluster_indices, cluster_indices)]
        avg_distances = np.mean(cluster_distances, axis=1)
        medoid_idx = cluster_indices[np.argmin(avg_distances)]

        cluster_medoids.append(medoid_idx)

        # Compute dispersion (average intra-cluster distance)
        dispersion = np.mean(cluster_distances)
        cluster_dispersions.append(dispersion)

    # Compute Davies-Bouldin index
    db_index = 0

    for i in range(len(cluster_medoids)):
        max_similarity = -np.inf

        for j in range(len(cluster_medoids)):
            if i == j:
                continue

            # Distance between cluster medoids
            d_ij = distance_matrix[cluster_medoids[i], cluster_medoids[j]]

            # Similarity measure
            similarity = (cluster_dispersions[i] + cluster_dispersions[j]) / d_ij

            if similarity > max_similarity:
                max_similarity = similarity

        db_index += max_similarity

    return db_index / len(cluster_medoids)