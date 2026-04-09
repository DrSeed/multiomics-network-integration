#!/usr/bin/env python3
"""
Multi-omics integration using Similarity Network Fusion.
Combines transcriptomics, methylation, and proteomics for patient stratification.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, squareform
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse
from pathlib import Path


def compute_affinity_matrix(X: np.ndarray, k: int = 20, mu: float = 0.5) -> np.ndarray:
    """Compute scaled exponential similarity (affinity) matrix."""
    D = squareform(pdist(X, metric="euclidean"))

    # Local scaling: average distance to k nearest neighbours
    sorted_D = np.sort(D, axis=1)
    epsilon = np.mean(sorted_D[:, 1:k+1], axis=1)

    W = np.zeros_like(D)
    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            if i != j:
                W[i, j] = np.exp(-D[i, j]**2 / (mu * epsilon[i] * epsilon[j]))

    return W


def compute_knn_matrix(W: np.ndarray, k: int = 20) -> np.ndarray:
    """Compute K-nearest neighbour matrix from affinity matrix."""
    n = W.shape[0]
    S = np.zeros_like(W)

    for i in range(n):
        neighbours = np.argsort(W[i])[::-1][:k]
        S[i, neighbours] = W[i, neighbours] / np.sum(W[i, neighbours])

    return S


def normalise_matrix(W: np.ndarray) -> np.ndarray:
    """Row-normalise a matrix."""
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return W / row_sums


def snf(affinity_matrices: list, k: int = 20, iterations: int = 20) -> np.ndarray:
    """
    Similarity Network Fusion.
    Fuses multiple patient similarity networks into a unified network.
    """
    n_views = len(affinity_matrices)
    n_samples = affinity_matrices[0].shape[0]

    # Compute normalised weight matrices and KNN matrices
    P = [normalise_matrix(W) for W in affinity_matrices]
    S = [compute_knn_matrix(W, k) for W in affinity_matrices]

    # Iterative fusion
    for t in range(iterations):
        P_new = []
        for v in range(n_views):
            # Average of all other views' normalised matrices
            others = [P[j] for j in range(n_views) if j != v]
            avg_others = np.mean(others, axis=0)

            # Update: P_v = S_v x avg(P_others) x S_v^T
            P_updated = S[v] @ avg_others @ S[v].T

            # Normalise
            P_updated = normalise_matrix(P_updated)
            P_new.append(P_updated)

        P = P_new

    # Final fused matrix: average of all views
    fused = np.mean(P, axis=0)
    fused = (fused + fused.T) / 2

    return fused


def cluster_patients(fused_matrix: np.ndarray, max_k: int = 6) -> tuple:
    """Determine optimal clusters using spectral clustering."""
    best_k, best_score = 2, -1

    for k in range(2, max_k + 1):
        sc = SpectralClustering(n_clusters=k, affinity="precomputed", random_state=42)
        labels = sc.fit_predict(fused_matrix)
        score = silhouette_score(fused_matrix, labels, metric="precomputed")

        if score > best_score:
            best_k, best_score = k, score

    # Final clustering with best k
    sc = SpectralClustering(n_clusters=best_k, affinity="precomputed", random_state=42)
    labels = sc.fit_predict(fused_matrix)

    return labels, best_k, best_score


def visualise_network(fused_matrix: np.ndarray, labels: np.ndarray, output_path: str):
    """Visualise fused network using MDS embedding."""
    from sklearn.manifold import MDS

    # Convert similarity to distance
    D = 1 - fused_matrix / fused_matrix.max()
    np.fill_diagonal(D, 0)

    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
    coords = mds.fit_transform(D)

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="Set2",
                         s=80, edgecolors="black", linewidth=0.5)
    ax.legend(*scatter.legend_elements(), title="Cluster")
    ax.set_title("Patient Stratification (SNF + Spectral Clustering)")
    ax.set_xlabel("MDS Dimension 1")
    ax.set_ylabel("MDS Dimension 2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Multi-omics SNF Integration")
    parser.add_argument("--expr", required=True, help="Expression matrix CSV")
    parser.add_argument("--methyl", required=True, help="Methylation matrix CSV")
    parser.add_argument("--protein", required=True, help="Proteomics matrix CSV")
    parser.add_argument("--output", default="results")
    parser.add_argument("--k_neighbours", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load omics data
    print("Loading multi-omics data...")
    expr = pd.read_csv(args.expr, index_col=0)
    methyl = pd.read_csv(args.methyl, index_col=0)
    protein = pd.read_csv(args.protein, index_col=0)

    # Align samples
    common = expr.index.intersection(methyl.index).intersection(protein.index)
    print(f"Common samples: {len(common)}")

    datasets = [
        StandardScaler().fit_transform(expr.loc[common].values),
        StandardScaler().fit_transform(methyl.loc[common].values),
        StandardScaler().fit_transform(protein.loc[common].values),
    ]

    # Compute affinity matrices
    print("Computing patient similarity networks...")
    affinity_matrices = [compute_affinity_matrix(X, k=args.k_neighbours) for X in datasets]

    # SNF
    print(f"Running SNF ({args.iterations} iterations)...")
    fused = snf(affinity_matrices, k=args.k_neighbours, iterations=args.iterations)

    # Clustering
    labels, n_clusters, sil_score = cluster_patients(fused)
    print(f"Optimal clusters: {n_clusters} (silhouette: {sil_score:.3f})")

    # Save results
    results_df = pd.DataFrame({"sample": common, "cluster": labels})
    results_df.to_csv(output_dir / "patient_clusters.csv", index=False)
    np.save(output_dir / "fused_network.npy", fused)

    # Visualise
    visualise_network(fused, labels, str(output_dir / "patient_stratification.png"))
    print("Integration complete.")


if __name__ == "__main__":
    main()
