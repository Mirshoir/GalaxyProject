"""
RMS vs k analysis for elbow method.
Computes RMS distance within clusters as function of k.
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from datetime import datetime

from src.compression import NCDCalculator
from src.preprocess import GalaxyDataset
from src.nearest_neighbors import NCDNearestNeighbors
from src.metrics import compute_rms_vs_k
from src.clustering import DistanceSpaceClustering

def load_or_compute_distances(data_dir="data/raw",
                            max_images=100,
                            cache_file="results/distances/distance_matrix.npy"):
    """
    Load precomputed distances or compute them.

    Args:
        data_dir: Directory with galaxy images
        max_images: Maximum number of images to process
        cache_file: Path to cache file for distance matrix

    Returns:
        Tuple of (distance_matrix, image_paths)
    """
    # Check if cached distances exist
    if os.path.exists(cache_file):
        print(f"Loading cached distances from {cache_file}")
        distance_matrix = np.load(cache_file)

        # Load corresponding image paths
        paths_file = cache_file.replace(".npy", "_paths.json")
        with open(paths_file, 'r') as f:
            image_paths = json.load(f)

        print(f"Loaded distance matrix for {len(image_paths)} images")
        return distance_matrix, image_paths

    # Compute distances
    print("Computing distances...")
    dataset = GalaxyDataset(data_dir)

    try:
        dataset.load_images("*.jpg")
    except:
        # Try other extensions
        for ext in ["*.png", "*.jpeg", "*.tiff"]:
            dataset.load_images(ext)
            if dataset.image_paths:
                break

    if not dataset.image_paths:
        raise ValueError(f"No images found in {data_dir}")

    # Limit number of images if specified
    if max_images and len(dataset.image_paths) > max_images:
        print(f"Limiting to first {max_images} images")
        dataset.image_paths = dataset.image_paths[:max_images]

    # Compute distance matrix
    neighbors = NCDNearestNeighbors()
    neighbors.fit(dataset.image_paths, compute_matrix=True, verbose=True)

    distance_matrix = neighbors.distance_matrix
    image_paths = dataset.image_paths

    # Cache results
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    np.save(cache_file, distance_matrix)

    # Save image paths
    paths_file = cache_file.replace(".npy", "_paths.json")
    with open(paths_file, 'w') as f:
        json.dump(image_paths, f)

    print(f"Saved distance matrix to {cache_file}")

    return distance_matrix, image_paths

def run_rms_vs_k_analysis(distance_matrix,
                         k_values=None,
                         n_runs=5,
                         random_state=42):
    """
    Main RMS vs k analysis.

    Args:
        distance_matrix: Precomputed N x N distance matrix
        k_values: Range of k values to test
        n_runs: Number of random initializations per k
        random_state: Random seed

    Returns:
        Dictionary with all results
    """
    if k_values is None:
        # Determine reasonable k range based on dataset size
        n_samples = distance_matrix.shape[0]
        max_k = min(20, n_samples // 5)  # At least 5 points per cluster
        k_values = list(range(2, max_k + 1))

    print(f"Running RMS vs k analysis for k = {k_values}")
    print(f"Dataset size: {distance_matrix.shape[0]} images")
    print(f"Number of runs per k: {n_runs}")

    # Compute RMS vs k
    k_array, mean_rms, std_rms = compute_rms_vs_k(
        distance_matrix, k_values, n_runs, random_state
    )

    # Fit curve to help identify elbow
    from scipy.optimize import curve_fit

    def exponential_decay(x, a, b, c):
        return a * np.exp(-b * x) + c

    try:
        # Try to fit exponential decay
        popt, _ = curve_fit(exponential_decay, k_array, mean_rms,
                           p0=[max(mean_rms), 0.1, min(mean_rms)])

        # Compute second derivative to find elbow
        fitted = exponential_decay(k_array, *popt)
        second_deriv = np.gradient(np.gradient(fitted))

        # Find point where second derivative is maximum (sharpest bend)
        elbow_idx = np.argmax(np.abs(second_deriv))
        elbow_k = k_array[elbow_idx]

    except:
        elbow_k = None
        fitted = None
        second_deriv = None

    # Package results
    results = {
        'k_values': k_array.tolist(),
        'mean_rms': mean_rms.tolist(),
        'std_rms': std_rms.tolist(),
        'elbow_k': int(elbow_k) if elbow_k is not None else None,
        'timestamp': datetime.now().isoformat(),
        'n_samples': distance_matrix.shape[0],
        'n_runs': n_runs,
        'random_state': random_state
    }

    return results, fitted, second_deriv

def visualize_rms_vs_k(results, fitted=None, second_deriv=None):
    """Create visualization of RMS vs k analysis."""
    k_values = results['k_values']
    mean_rms = results['mean_rms']
    std_rms = results['std_rms']
    elbow_k = results['elbow_k']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: RMS vs k with error bars
    axes[0].errorbar(k_values, mean_rms, yerr=std_rms,
                    fmt='o-', capsize=5, capthick=2, linewidth=2)
    axes[0].set_xlabel('Number of Clusters (k)')
    axes[0].set_ylabel('Average RMS Distance')
    axes[0].set_title('RMS Distance vs Number of Clusters')
    axes[0].grid(True, alpha=0.3)

    # Highlight elbow point if found
    if elbow_k is not None:
        idx = k_values.index(elbow_k)
        axes[0].scatter([elbow_k], [mean_rms[idx]],
                       color='red', s=200, zorder=5,
                       label=f'Elbow at k={elbow_k}')
        axes[0].axvline(x=elbow_k, color='red', linestyle='--', alpha=0.5)
        axes[0].legend()

    # Plot 2: Rate of change (first derivative)
    rates = -np.diff(mean_rms) / np.diff(k_values)
    axes[1].plot(k_values[1:], rates, 's-', linewidth=2)
    axes[1].set_xlabel('Number of Clusters (k)')
    axes[1].set_ylabel('Rate of RMS Decrease')
    axes[1].set_title('Rate of Improvement')
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Second derivative (if available)
    if second_deriv is not None:
        axes[2].plot(k_values, second_deriv, '^-', linewidth=2)
        axes[2].set_xlabel('Number of Clusters (k)')
        axes[2].set_ylabel('Second Derivative')
        axes[2].set_title('Curvature Analysis')
        axes[2].grid(True, alpha=0.3)

        if elbow_k is not None:
            axes[2].axvline(x=elbow_k, color='red', linestyle='--', alpha=0.5)
    else:
        # Alternative: Log plot of RMS
        axes[2].semilogy(k_values, mean_rms, 'o-', linewidth=2)
        axes[2].set_xlabel('Number of Clusters (k)')
        axes[2].set_ylabel('Log(RMS Distance)')
        axes[2].set_title('Log Scale View')
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    os.makedirs("results/figures", exist_ok=True)
    plt.savefig("results/figures/rms_vs_k_analysis.png", dpi=150, bbox_inches='tight')
    plt.savefig("results/figures/rms_vs_k_analysis.pdf", bbox_inches='tight')

    plt.show()

def analyze_cluster_quality(distance_matrix, k_values=None):
    """Analyze cluster quality metrics for different k."""
    if k_values is None:
        k_values = [2, 3, 4, 5, 6, 7, 8]

    from src.metrics import compute_silhouette_score, compute_davies_bouldin_index

    results = []

    for k in tqdm(k_values, desc="Analyzing cluster quality"):
        # Cluster with this k
        clusterer = DistanceSpaceClustering(n_clusters=k, random_state=42)
        clusterer.fit(distance_matrix)

        # Compute metrics
        silhouette = compute_silhouette_score(distance_matrix, clusterer.labels_)
        db_index = compute_davies_bouldin_index(distance_matrix, clusterer.labels_)

        # Get cluster sizes
        from collections import Counter
        cluster_sizes = Counter(clusterer.labels_)

        results.append({
            'k': k,
            'silhouette': float(silhouette),
            'davies_bouldin': float(db_index),
            'cluster_sizes': dict(cluster_sizes),
            'n_clusters': len(cluster_sizes)
        })

    return results

def main():
    """Main entry point for RMS vs k analysis."""
    print("=" * 60)
    print("RMS vs k Analysis for Galaxy Morphology Clustering")
    print("=" * 60)

    # Load or compute distances
    try:
        distance_matrix, image_paths = load_or_compute_distances(
            max_images=100  # Adjust based on your dataset size
        )
    except Exception as e:
        print(f"Error loading/computing distances: {e}")
        print("\nPlease add some galaxy images to data/raw/")
        print("Supported formats: .jpg, .png, .jpeg, .tiff")
        return

    # Define k range based on dataset size
    n_samples = distance_matrix.shape[0]
    if n_samples < 10:
        k_values = list(range(2, min(6, n_samples)))
    elif n_samples < 50:
        k_values = list(range(2, min(10, n_samples // 2)))
    else:
        k_values = list(range(2, min(20, n_samples // 5)))

    print(f"\nDataset: {n_samples} galaxies")
    print(f"Testing k values: {k_values}")

    # Run RMS vs k analysis
    results, fitted, second_deriv = run_rms_vs_k_analysis(
        distance_matrix, k_values, n_runs=5, random_state=42
    )

    # Visualize results
    visualize_rms_vs_k(results, fitted, second_deriv)

    # Save numerical results
    os.makedirs("results/rms", exist_ok=True)

    results_file = "results/rms/rms_vs_k_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {results_file}")

    # Optional: Additional cluster quality analysis
    print("\n" + "=" * 60)
    print("Additional Cluster Quality Analysis")
    print("=" * 60)

    quality_results = analyze_cluster_quality(distance_matrix, k_values[:min(8, len(k_values))])

    # Save quality results
    quality_file = "results/rms/cluster_quality_results.json"
    with open(quality_file, 'w') as f:
        json.dump(quality_results, f, indent=2)

    # Print summary
    print("\nCluster Quality Summary:")
    print("-" * 40)
    for result in quality_results:
        print(f"k={result['k']:2d}: "
              f"Silhouette={result['silhouette']:.3f}, "
              f"DB Index={result['davies_bouldin']:.3f}, "
              f"Clusters={result['n_clusters']}")

    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Examine the elbow plot in results/figures/rms_vs_k_analysis.png")
    print("2. Choose optimal k based on elbow point and quality metrics")
    print("3. Run clustering with chosen k for detailed morphology analysis")

if __name__ == "__main__":
    main()