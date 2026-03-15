from __future__ import annotations

import numpy as np


def compute_distance_matrix(coords: np.ndarray, edge_weight_type: str = "EUC_2D") -> np.ndarray:
    """Compute a TSPLIB-style distance matrix using NumPy."""
    diff = coords[:, None, :] - coords[None, :, :]
    euclidean = np.sqrt(np.sum(diff * diff, axis=2))

    if edge_weight_type == "EUC_2D":
        dist = np.rint(euclidean)
    elif edge_weight_type == "CEIL_2D":
        dist = np.ceil(euclidean)
    else:
        raise ValueError(f"Unsupported edge_weight_type: {edge_weight_type}")

    np.fill_diagonal(dist, 0.0)
    return dist.astype(np.float64)


def build_candidate_lists(dist_matrix: np.ndarray, k: int) -> np.ndarray:
    """Return the k nearest neighbors for each city."""
    n = dist_matrix.shape[0]
    k = max(1, min(k, n - 1))
    order = np.argsort(dist_matrix, axis=1)
    candidates = []
    for i in range(n):
        row = order[i]
        row = row[row != i][:k]
        candidates.append(row)
    return np.asarray(candidates, dtype=np.int32)
