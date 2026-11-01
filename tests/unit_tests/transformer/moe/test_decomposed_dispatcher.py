import numpy as np
import pytest
import torch

from megatron.core.transformer.moe.token_dispatcher import MoEDecomposedAlltoAllDispatcher
from megatron.core.typed_torch import apply_module
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.moe.test_token_dispatcher import MoEModelTestContainer


def _all_gather_cpu(array):
    """All-gather a small integer array from every rank (default group), as numpy."""
    tensor = torch.as_tensor(np.asarray(array, dtype=np.int64)).cuda()
    gathered = [torch.empty_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(gathered, tensor)
    return [g.cpu().numpy() for g in gathered]


@pytest.mark.skipif(Utils.world_size < 2, reason="decomposed_alltoall needs EP > 1")
class TestDecomposedDispatcherPreprocess:
    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("num_local_experts", [1, 2, 4])
    @pytest.mark.parametrize("moe_permute_fusion", [False, True])
    @pytest.mark.parametrize("coalesce_tokens", [0, 64, None])
    def test_preprocess_schedule(self, num_local_experts, moe_permute_fusion, coalesce_tokens):
        ep_size = Utils.world_size
        container = MoEModelTestContainer(
            tp_size=1,
            ep_size=ep_size,
            pp_size=1,
            num_moe_experts=ep_size * num_local_experts,
            moe_router_topk=2,
            moe_router_load_balancing_type="aux_loss",
            moe_token_dispatcher_type="decomposed_alltoall",
            moe_permute_fusion=moe_permute_fusion,
        )
        moe_layer = container.moe_layer
        dispatcher = moe_layer.token_dispatcher
        # The test container does not forward this config field; set it on the dispatcher.
        dispatcher.coalesce_tokens = coalesce_tokens
        assert isinstance(dispatcher, MoEDecomposedAlltoAllDispatcher)

        rank = torch.distributed.get_rank()
        # Different token counts and routing per rank, so D is not uniform.
        torch.manual_seed(1000 + rank)
        num_tokens = 16 * (rank + 1)
        hidden_states = torch.randn(
            (num_tokens, 1, moe_layer.config.hidden_size), dtype=torch.float32
        ).cuda()
        probs, routing_map = apply_module(moe_layer.router)(hidden_states)
        dispatcher.dispatch_preprocess(hidden_states, routing_map, probs)

        n, me = ep_size, dispatcher.ep_rank
        demand = dispatcher.demand

        # The same demand matrix and schedule on every rank.
        for other in _all_gather_cpu(demand):
            np.testing.assert_array_equal(other, demand)
        schedule = np.stack(dispatcher.schedule)
        for other in _all_gather_cpu(schedule):
            np.testing.assert_array_equal(other, schedule)

        # This rank's row and column match the stock splits.
        np.testing.assert_array_equal(demand[me, :], np.asarray(dispatcher.input_splits))
        np.testing.assert_array_equal(demand[:, me], np.asarray(dispatcher.output_splits))
        # Every routed token appears once in D (top-2 routing).
        assert demand.sum() == 2 * sum(16 * (r + 1) for r in range(n))

        # Per-expert counts add up to the stock tokens per local expert.
        np.testing.assert_array_equal(
            dispatcher.tokens_per_src_expert.sum(axis=0),
            torch.as_tensor(dispatcher.tokens_per_expert).cpu().numpy(),
        )
        np.testing.assert_array_equal(dispatcher.tokens_per_src_expert.sum(axis=1), demand[:, me])

        # n - 1 matchings; receive chunk 0 is local, the rest one per matching, every source once.
        assert len(dispatcher.schedule) == n - 1
        assert dispatcher.recv_src[0] == me
        assert sorted(dispatcher.recv_src) == list(range(n))
        for k, perm in enumerate(dispatcher.schedule, start=1):
            assert perm[dispatcher.recv_src[k]] == me
        assert dispatcher.recv_counts == [int(demand[src, me]) for src in dispatcher.recv_src]
        assert dispatcher.recv_offsets[0] == 0
        assert dispatcher.recv_offsets[-1] + dispatcher.recv_counts[-1] == demand[:, me].sum()
        assert dispatcher.send_offsets[me] + dispatcher.input_splits[me] <= demand[me, :].sum()

        # Batches cover every chunk once, in order.
        covered = [k for first, last in dispatcher.batches for k in range(first, last + 1)]
        assert covered == list(range(n))
        assert len(dispatcher.batch_of_chunk) == n
        if coalesce_tokens is None:
            assert dispatcher.batches == [(0, n - 1)]
        elif coalesce_tokens == 0:
            assert dispatcher.batches == [(k, k) for k in range(n)]


@pytest.mark.skipif(Utils.world_size < 2, reason="decomposed_alltoall needs EP > 1")
class TestDecomposedDispatcherForward:
    """Full MoE-layer forward: decomposed vs stock alltoall, identical weights and inputs."""

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def _compare(self, num_local_experts, coalesce_tokens, moe_grouped_gemm, tokens_for_rank):
        ep_size = Utils.world_size
        container = MoEModelTestContainer(
            tp_size=1,
            ep_size=ep_size,
            pp_size=1,
            num_moe_experts=ep_size * num_local_experts,
            moe_router_topk=2,
            moe_router_load_balancing_type="none",
            moe_aux_loss_coeff=0.0,
            moe_token_dispatcher_type="alltoall",
            moe_grouped_gemm=moe_grouped_gemm,
            hidden_size=64,
            test_dtype=torch.bfloat16,
        )
        stock = container.moe_layer.eval()
        decomposed = container.new_moe_layer(moe_token_dispatcher_type="decomposed_alltoall")
        decomposed.load_state_dict(stock.state_dict())
        decomposed.eval()
        dispatcher = decomposed.token_dispatcher
        assert isinstance(dispatcher, MoEDecomposedAlltoAllDispatcher)
        dispatcher.coalesce_tokens = coalesce_tokens

        rank = torch.distributed.get_rank()
        torch.manual_seed(1000 + rank)
        hidden_states = torch.randn(
            (tokens_for_rank(rank), 1, stock.config.hidden_size), dtype=torch.bfloat16
        ).cuda()
        with torch.no_grad():
            expected, _ = stock(hidden_states)
            # Twice: the second call reuses the shared streams and communicators.
            for _ in range(2):
                actual, _ = decomposed(hidden_states)
        torch.cuda.synchronize()
        return expected, actual

    @pytest.mark.parametrize("moe_grouped_gemm", [False, True])
    @pytest.mark.parametrize("num_local_experts", [1, 2, 4])
    def test_one_batch_matches_stock_bitwise(self, num_local_experts, moe_grouped_gemm):
        # tau=None: one expert call over all received tokens, same per-expert counts as stock.
        expected, actual = self._compare(
            num_local_experts, None, moe_grouped_gemm, lambda rank: 16 * (rank + 1)
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    @pytest.mark.parametrize("moe_grouped_gemm", [False, True])
    @pytest.mark.parametrize("num_local_experts", [1, 2, 4])
    @pytest.mark.parametrize("coalesce_tokens", [0, 48])
    def test_pipelined_matches_stock(self, coalesce_tokens, num_local_experts, moe_grouped_gemm):
        # Smaller expert calls may pick other GEMM kernels, so allclose rather than bitwise.
        expected, actual = self._compare(
            num_local_experts, coalesce_tokens, moe_grouped_gemm, lambda rank: 16 * (rank + 1)
        )
        torch.testing.assert_close(actual, expected)

    @pytest.mark.parametrize("coalesce_tokens", [0, None])
    def test_sparse_and_skewed_demand(self, coalesce_tokens):
        # One token on most ranks: most pairs carry 0 tokens, whole matchings are empty, and
        # ranks send nothing in a matching while still receiving (and the reverse).
        expected, actual = self._compare(
            4, coalesce_tokens, False, lambda rank: 64 if rank == 0 else 1
        )
        torch.testing.assert_close(actual, expected)
