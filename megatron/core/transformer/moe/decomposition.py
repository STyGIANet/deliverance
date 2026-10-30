"""Decomposition of the MoE token demand matrix into matchings for the
'decomposed_alltoall' token dispatcher.

All functions run on the CPU with numpy arrays and are deterministic, so every EP rank derives the
same schedule from the same demand matrix.
"""

from typing import Callable, Dict, List

import numpy as np


def greedy_maxweight_decompose(weights: np.ndarray) -> List[np.ndarray]:
    """Greedily decompose the off-diagonal complete bipartite graph into `n - 1` perfect matchings.

    Each round picks the available perfect matching with maximum total weight under the original
    edge weights (Jonker-Volgenant, `scipy.optimize.linear_sum_assignment`), then removes those
    matched edges from the graph, whatever their weight. The diagonal is never available: local
    tokens are not routed. The residual graph stays regular, so every round finds a perfect
    matching and every off-diagonal `(src, dst)` pair appears in exactly one matching.

    Args:
        weights: [n, n] non-negative weights, `weights[src, dst]` (by default the token counts).

    Returns:
        `n - 1` permutations `perm` (int64 arrays of length n) in decomposition order. In a
        matching, rank `i` sends to `perm[i]`; `perm[i] != i`.
    """
    # Only the decomposed dispatcher needs scipy.
    from scipy.optimize import linear_sum_assignment

    weights = np.asarray(weights, dtype=np.int64)
    if weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
        raise ValueError("maxweight decomposition requires a square matrix.")

    n = weights.shape[0]
    if n <= 1:
        return []

    available = np.ones((n, n), dtype=bool)
    np.fill_diagonal(available, False)

    max_weight = float(np.max(weights)) if weights.size else 0.0
    blocked_cost = max_weight + 1.0
    matchings: List[np.ndarray] = []

    while available.any():
        cost = np.where(available, -weights.astype(np.float64), blocked_cost)
        row_ind, col_ind = linear_sum_assignment(cost)
        if len(row_ind) != n or np.any(~available[row_ind, col_ind]):
            raise ValueError("Failed to find a valid perfect matching in maxweight decomposition.")

        matchings.append(col_ind.astype(np.int64))

        available[row_ind, col_ind] = False

    return matchings


def order_largest_first(matchings: List[np.ndarray], demand: np.ndarray) -> List[np.ndarray]:
    """Sort matchings by their largest pair `demand[i, perm[i]]`, largest first (stable)."""
    rows = np.arange(demand.shape[0])
    return sorted(matchings, key=lambda perm: -int(demand[rows, perm].max()))


def order_as_is(matchings: List[np.ndarray], demand: np.ndarray) -> List[np.ndarray]:
    """Keep the decomposition order."""
    return list(matchings)


DECOMPOSE_FNS: Dict[str, Callable[[np.ndarray], List[np.ndarray]]] = {
    "greedy_mw": greedy_maxweight_decompose
}

ORDER_FNS: Dict[str, Callable[[List[np.ndarray], np.ndarray], List[np.ndarray]]] = {
    "largest_first": order_largest_first,
    "as_is": order_as_is,
}
