import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List

rng = np.random.default_rng(0)
def rounds(W):
    R = W.copy(); k = 0; zero_pairs = 0; self_pairs = 0
    while R.any():
        r, c = linear_sum_assignment(R, maximize=True)
        zero_pairs += int((R[r, c] == 0).sum()); self_pairs += int((r == c).sum())
        R[r, c] = 0; k += 1
    return k, zero_pairs


def maxweight_match_decomposition(
    count_matrix: np.ndarray,
    bytes_per_token: int,
) -> List[np.ndarray]:
    """
    Greedily decompose the off-diagonal complete bipartite graph into `n - 1`
    perfect matchings. Each round picks the available perfect matching with
    maximum total weight under the original edge weights, then removes those
    matched edges from the graph. The returned execution order is sorted from
    largest to smallest round size.

    Communication per round is the largest off-diagonal matched amount in
    bytes. Compute per round is the largest matched amount in tokens.
    """
    if bytes_per_token <= 0:
        raise ValueError("bytes_per_token must be positive.")

    weights = np.asarray(count_matrix, dtype=np.int64)
    if weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
        raise ValueError("maxweight decomposition requires a square matrix.")

    n = weights.shape[0]
    if n <= 1:
        return []

    available = np.ones((n, n), dtype=bool)
    # np.fill_diagonal(available, False)

    max_weight = float(np.max(weights)) if weights.size else 0.0
    blocked_cost = max_weight + 1.0
    results: List[np.ndarray] = []

    while available.any():
        cost = np.where(available, -weights.astype(np.float64), blocked_cost)
        row_ind, col_ind = linear_sum_assignment(cost)
        if len(row_ind) != n or np.any(~available[row_ind, col_ind]):
            raise ValueError("Failed to find a valid perfect matching in maxweight decomposition.")

        round_amounts = np.zeros_like(weights)
        round_amounts[row_ind, col_ind] = weights[row_ind, col_ind]

        results.append(round_amounts)

        available[row_ind, col_ind] = False

    # Sorted is order()
    # return sorted(results, reverse=False)
    return results








for n in (3, 8):
    worst = 0; ex = None; zp = 0
    for _ in range(20000 if n == 3 else 3000):
        W = rng.integers(0, 1000, (n, n)) * (rng.random((n, n)) < 0.8)
        l = maxweight_match_decomposition(W, 1)
        print(len(l))
        k, z = rounds(W); zp += z
        if k > worst: worst, ex = k, W
    print(f"n={n}: max rounds {worst}; zero-weight pairs inside matchings seen: {zp}")
    if worst > n: print(ex)
