import pytest

from megatron.core.transformer.transformer_config import TransformerConfig


def _make_decomposed_config(**overrides) -> TransformerConfig:
    kwargs = dict(
        num_layers=1,
        hidden_size=128,
        num_attention_heads=4,
        num_moe_experts=8,
        expert_model_parallel_size=8,
        moe_token_dispatcher_type="decomposed_alltoall",
        bf16=True,
    )
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


def test_decomposed_alltoall_defaults():
    config = _make_decomposed_config()

    assert config.moe_decomposed_coalesce_tokens == 0
    assert config.moe_decomposed_decompose == "greedy_mw"
    assert config.moe_decomposed_order == "largest_first"


@pytest.mark.parametrize("num_moe_experts", [8, 16, 64])
def test_decomposed_alltoall_accepts_multiple_local_experts(num_moe_experts):
    config = _make_decomposed_config(num_moe_experts=num_moe_experts)

    assert config.num_moe_experts // config.expert_model_parallel_size == num_moe_experts // 8


@pytest.mark.parametrize("coalesce_tokens", [None, 0, 512])
def test_decomposed_alltoall_accepts_coalesce_tokens(coalesce_tokens):
    config = _make_decomposed_config(moe_decomposed_coalesce_tokens=coalesce_tokens)

    assert config.moe_decomposed_coalesce_tokens == coalesce_tokens


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        pytest.param(
            {"num_moe_experts": None, "expert_model_parallel_size": 1},
            "requires num_moe_experts",
            id="no-moe",
        ),
        pytest.param(
            {"num_moe_experts": 1, "expert_model_parallel_size": 1},
            "expert_model_parallel_size > 1",
            id="ep1",
        ),
        pytest.param(
            {"tensor_model_parallel_size": 2, "sequence_parallel": True},
            "tensor_model_parallel_size == 1",
            id="tp2",
        ),
        pytest.param(
            {"expert_tensor_parallel_size": 2}, "expert_tensor_parallel_size == 1", id="etp2"
        ),
        pytest.param(
            {"cuda_graph_impl": "local"}, "does not support CUDA graphs", id="cuda-graphs"
        ),
        pytest.param(
            {"moe_expert_capacity_factor": 1.0}, "requires dropless routing", id="capacity-factor"
        ),
        pytest.param(
            {"transformer_impl": "inference_optimized"}, "inference_optimized", id="inference-opt"
        ),
        pytest.param(
            {"moe_shared_expert_intermediate_size": 128, "moe_shared_expert_overlap": True},
            "moe_shared_expert_overlap",
            id="shared-expert-overlap",
        ),
        pytest.param(
            {"moe_decomposed_coalesce_tokens": -1}, "must be None or >= 0", id="negative-tau"
        ),
    ],
)
def test_decomposed_alltoall_rejects_unsupported_configurations(overrides, message):
    with pytest.raises(ValueError, match=message):
        _make_decomposed_config(**overrides)
