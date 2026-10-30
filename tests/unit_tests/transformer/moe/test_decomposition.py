import numpy as np
import pytest

from megatron.core.transformer.moe.decomposition import (
    DECOMPOSE_FNS,
    ORDER_FNS,
    greedy_maxweight_decompose,
    order_as_is,
    order_largest_first,
)


def _random_demand(n, density, seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 1000, (n, n)) * (rng.random((n, n)) < density)


def _check_decomposition(demand, matchings):
    n = demand.shape[0]
    assert len(matchings) == n - 1
    used = np.zeros((n, n), dtype=np.int64)
    for perm in matchings:
        assert sorted(perm.tolist()) == list(range(n)), "not a permutation"
        assert np.all(perm != np.arange(n)), "matching uses the diagonal"
        used[np.arange(n), perm] += 1
    expected = np.ones((n, n), dtype=np.int64)
    np.fill_diagonal(expected, 0)
    assert np.array_equal(used, expected), "each off-diagonal pair must be in exactly one matching"


@pytest.mark.parametrize("n", [2, 3, 4, 8, 16])
@pytest.mark.parametrize("density", [0.0, 0.3, 0.8, 1.0])
def test_decomposition_covers_each_pair_once(n, density):
    for seed in range(20):
        demand = _random_demand(n, density, seed)
        _check_decomposition(demand, greedy_maxweight_decompose(demand))


def test_decomposition_hand_checked():
    # Off-diagonal: the two derangements of 3 ranks weigh 5+7+9=21 and 1+2+3=6.
    demand = np.array([[0, 5, 1], [2, 0, 7], [9, 3, 0]])
    matchings = greedy_maxweight_decompose(demand)
    assert [m.tolist() for m in matchings] == [[1, 2, 0], [2, 0, 1]]


def test_decomposition_ignores_diagonal():
    demand = np.diag([100, 200, 300, 400])
    _check_decomposition(demand, greedy_maxweight_decompose(demand))


def test_decomposition_is_deterministic():
    demand = _random_demand(8, 0.5, seed=0)
    first = greedy_maxweight_decompose(demand)
    for _ in range(5):
        again = greedy_maxweight_decompose(demand.copy())
        assert all(np.array_equal(a, b) for a, b in zip(first, again))


def test_decomposition_rounds_are_greedy_max_weight():
    # Each round is the heaviest perfect matching among the pairs not used by earlier rounds.
    from itertools import permutations

    n = 5
    demand = _random_demand(n, 0.7, seed=3)
    used = np.eye(n, dtype=bool)
    for perm in greedy_maxweight_decompose(demand):
        best = max(
            demand[np.arange(n), list(p)].sum()
            for p in permutations(range(n))
            if not used[np.arange(n), list(p)].any()
        )
        assert demand[np.arange(n), perm].sum() == best
        used[np.arange(n), perm] = True


@pytest.mark.parametrize("n", [0, 1])
def test_decomposition_trivial_sizes(n):
    assert greedy_maxweight_decompose(np.zeros((n, n))) == []


def test_decomposition_rejects_non_square():
    with pytest.raises(ValueError, match="square"):
        greedy_maxweight_decompose(np.zeros((2, 3)))


def test_order_largest_first():
    demand = np.array([[0, 5, 1], [2, 0, 7], [9, 3, 0]])
    small, large = np.array([2, 0, 1]), np.array([1, 2, 0])
    assert [m.tolist() for m in order_largest_first([small, large], demand)] == [
        large.tolist(),
        small.tolist(),
    ]


def test_order_largest_first_is_stable():
    demand = np.ones((3, 3), dtype=np.int64)
    a, b = np.array([1, 2, 0]), np.array([2, 0, 1])
    assert [m.tolist() for m in order_largest_first([a, b], demand)] == [a.tolist(), b.tolist()]
    assert [m.tolist() for m in order_largest_first([b, a], demand)] == [b.tolist(), a.tolist()]


def test_order_as_is():
    demand = np.zeros((3, 3), dtype=np.int64)
    a, b = np.array([1, 2, 0]), np.array([2, 0, 1])
    assert order_as_is([b, a], demand) == [b, a]


def test_registries_match_config_choices():
    import typing

    from megatron.core.transformer.transformer_config import TransformerConfig

    hints = typing.get_type_hints(TransformerConfig)
    assert set(DECOMPOSE_FNS) == set(typing.get_args(hints["moe_decomposed_decompose"]))
    assert set(ORDER_FNS) == set(typing.get_args(hints["moe_decomposed_order"]))
