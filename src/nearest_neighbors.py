"""
Nearest neighbors search using NCD distances.
Implements efficient k-NN search in distance space.
"""
import numpy as np
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm
import heapq
from .compression import NCDCalculator


class NCDNearestNeighbors:
    """k-NN search using NCD distances."""

    def __init__(self, calculator: NCDCalculator = None):
        self.calculator = calculator or NCDCalculator()
        self.image_paths = []
        self.distance_matrix = None

    def fit(self, image_paths: List[str],
            compute_matrix: bool = True,
            verbose: bool = True) -> 'NCDNearestNeighbors':
        """
        Prepare the nearest neighbors search.

        Args:
            image_paths: List of image file paths
            compute_matrix: If True, precompute all pairwise distances
            verbose: Show progress bar
        """
        self.image_paths = image_paths

        if compute_matrix:
            n = len(image_paths)
            self.distance_matrix = np.zeros((n, n))

            if verbose:
                print(f"Computing distance matrix for {n} images...")
                iterator = tqdm(range(n), desc="Computing distances")
            else:
                iterator = range(n)

            # Compute pairwise distances (only upper triangle)
            for i in iterator:
                for j in range(i, n):
                    if i == j:
                        self.distance_matrix[i, j] = 0.0
                    else:
                        dist = self.calculator.ncd(
                            image_paths[i],
                            image_paths[j]
                        )
                        self.distance_matrix[i, j] = dist
                        self.distance_matrix[j, i] = dist  # Symmetric

        return self

    def kneighbors(self,
                   query_idx: int,
                   k: int = 5,
                   use_precomputed: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find k nearest neighbors for a query image.

        Args:
            query_idx: Index of query image in self.image_paths
            k: Number of neighbors to find
            use_precomputed: Use precomputed distance matrix if available

        Returns:
            Tuple of (distances, indices) for k nearest neighbors
        """
        if use_precomputed and self.distance_matrix is not None:
            # Use precomputed distances
            distances = self.distance_matrix[query_idx]
            indices = np.argsort(distances)[1:k + 1]  # Skip self (distance=0)
            return distances[indices], indices

        else:
            # Compute distances on the fly
            distances = []

            for i, path in enumerate(self.image_paths):
                if i == query_idx:
                    distances.append(0.0)
                else:
                    dist = self.calculator.ncd(
                        self.image_paths[query_idx],
                        path
                    )
                    distances.append(dist)

            distances = np.array(distances)
            indices = np.argsort(distances)[1:k + 1]
            return distances[indices], indices

    def kneighbors_by_path(self,
                           query_path: str,
                           k: int = 5) -> Tuple[np.ndarray, List[str]]:
        """
        Find k nearest neighbors for a query image by path.

        Returns:
            Tuple of (distances, paths) for k nearest neighbors
        """
        # Add query to dataset temporarily
        all_paths = self.image_paths + [query_path]
        query_idx = len(self.image_paths)

        # Compute distances to all images
        distances = []
        for path in all_paths[:-1]:  # Skip the query itself
            dist = self.calculator.ncd(query_path, path)
            distances.append(dist)

        distances = np.array(distances)

        # Get k smallest distances
        if k < len(distances):
            idx = np.argpartition(distances, k)[:k]
            sorted_idx = idx[np.argsort(distances[idx])]
        else:
            sorted_idx = np.argsort(distances)

        return distances[sorted_idx], [self.image_paths[i] for i in sorted_idx]

    def get_distance_vector(self, image_idx: int) -> np.ndarray:
        """Get distance vector from one image to all others."""
        if self.distance_matrix is not None:
            return self.distance_matrix[image_idx]
        else:
            distances = []
            for i, path in enumerate(self.image_paths):
                if i == image_idx:
                    distances.append(0.0)
                else:
                    dist = self.calculator.ncd(
                        self.image_paths[image_idx],
                        path
                    )
                    distances.append(dist)
            return np.array(distances)


def find_similar_pairs(neighbors: NCDNearestNeighbors,
                       threshold: float = 0.1) -> List[Tuple[int, int, float]]:
    """
    Find all pairs of images with NCD distance below threshold.

    Returns:
        List of (idx1, idx2, distance) tuples
    """
    if neighbors.distance_matrix is None:
        raise ValueError("Distance matrix not computed. Call fit() with compute_matrix=True first.")

    n = len(neighbors.image_paths)
    similar_pairs = []

    for i in range(n):
        for j in range(i + 1, n):
            dist = neighbors.distance_matrix[i, j]
            if dist < threshold:
                similar_pairs.append((i, j, dist))

    # Sort by distance
    similar_pairs.sort(key=lambda x: x[2])
    return similar_pairs


def save_neighbors_results(distances: np.ndarray,
                           indices: np.ndarray,
                           output_file: str):
    """Save nearest neighbors results to file."""
    results = {
        'distances': distances.tolist(),
        'indices': indices.tolist()
    }

    import json
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)