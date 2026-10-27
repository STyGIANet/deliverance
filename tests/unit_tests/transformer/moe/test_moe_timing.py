import json

import pytest
import torch

import megatron.core.transformer.moe.moe_timing as moe_timing
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_submodules
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.moe.moe_timing import MoEPhaseTimer, moe_phase
from megatron.core.transformer.spec_utils import get_submodules
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.initialize import _set_random_seed
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _read_rows(out_dir, rank=0):
    with open(out_dir / f"rank{rank:05d}.jsonl") as f:
        return [json.loads(line) for line in f]


def _work(n=256):
    a = torch.randn(n, n, device="cuda")
    return a @ a


class TestMoEPhaseTimer:

    def test_rows_steps_and_sequential_intervals(self, tmp_path):
        timer = MoEPhaseTimer(str(tmp_path), rank=0)
        for _ in range(2):  # steps
            for layer in (1, 2):
                record = timer.begin_layer(layer=layer, path="Stock", n_tokens_local=8)
                for name in ("dispatch", "compute", "combine"):
                    with moe_phase(record, name, tokens=4):
                        _work()
                timer.end_layer(record)
        timer.close()

        rows = _read_rows(tmp_path)
        assert [(r["step"], r["layer"]) for r in rows] == [(0, 1), (0, 2), (1, 1), (1, 2)]
        for row in rows:
            phases = row["intervals"]
            assert [p["phase"] for p in phases] == ["dispatch", "compute", "combine"]
            assert all(p["tokens"] == 4 for p in phases)
            prev_end = 0.0
            for p in phases:  # one stream: intervals are ordered and non-overlapping
                assert prev_end <= p["start_us"] <= p["end_us"]
                prev_end = p["end_us"]

    def test_side_stream_interval_overlaps(self, tmp_path):
        timer = MoEPhaseTimer(str(tmp_path), rank=0)
        side = torch.cuda.Stream()
        record = timer.begin_layer(layer=1, path="Decomposed", n_tokens_local=8)
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            with moe_phase(record, "dispatch", idx=0):
                _work(2048)
        with moe_phase(record, "compute", idx=0):
            _work(2048)
        torch.cuda.current_stream().wait_stream(side)
        timer.end_layer(record)
        timer.close()

        (row,) = _read_rows(tmp_path)
        assert {(p["phase"], p["idx"]) for p in row["intervals"]} == {
            ("dispatch", 0),
            ("compute", 0),
        }
        assert all(0.0 <= p["start_us"] <= p["end_us"] for p in row["intervals"])

    def test_tokens_set_inside_block_and_failed_phase_dropped(self, tmp_path):
        timer = MoEPhaseTimer(str(tmp_path), rank=0)
        record = timer.begin_layer(layer=1, path="Stock", n_tokens_local=8)
        with moe_phase(record, "dispatch") as interval:
            interval.tokens = 7
        with pytest.raises(RuntimeError):
            with moe_phase(record, "compute"):
                raise RuntimeError
        timer.end_layer(record)
        timer.close()

        (row,) = _read_rows(tmp_path)
        assert [(p["phase"], p["tokens"]) for p in row["intervals"]] == [("dispatch", 7)]

    def test_disabled_phase_is_noop(self):
        with moe_phase(None, "dispatch") as interval:
            interval.tokens = 3  # must not raise

    def test_refuses_existing_output(self, tmp_path):
        MoEPhaseTimer(str(tmp_path), rank=0).close()
        with pytest.raises(FileExistsError):
            MoEPhaseTimer(str(tmp_path), rank=0)


class TestMoELayerTiming:

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_alltoall_forward_writes_stock_phases(self, tmp_path, monkeypatch):
        Utils.initialize_model_parallel(1, 1)
        _set_random_seed(seed_=123, data_parallel_random_init=False)
        out_dir = tmp_path / f"rank{torch.distributed.get_rank()}"
        monkeypatch.setenv(moe_timing.MOE_PHASE_TIMING_ENV, str(out_dir))
        monkeypatch.setattr(moe_timing, "_timer", None)
        monkeypatch.setattr(moe_timing, "_timer_checked", False)

        config = TransformerConfig(
            num_layers=1,
            hidden_size=64,
            num_attention_heads=4,
            num_moe_experts=4,
            moe_token_dispatcher_type="alltoall",
            moe_router_topk=2,
            moe_ffn_hidden_size=128,
            add_bias_linear=False,
            moe_grouped_gemm=False,
        )
        submodules = get_submodules(
            get_gpt_layer_local_submodules(num_experts=4, moe_grouped_gemm=False).mlp
        )
        assert isinstance(submodules, MoESubmodules)
        moe_layer = MoELayer(config, submodules, layer_number=1).cuda().eval()
        hidden_states = torch.randn(16, 2, 64, device="cuda")

        with torch.no_grad():
            for _ in range(2):
                moe_layer(hidden_states)
        moe_timing.get_moe_phase_timer().close()

        rows = _read_rows(out_dir, rank=torch.distributed.get_rank())
        assert [r["step"] for r in rows] == [0, 1]
        for row in rows:
            assert row["path"] == "MoEAlltoAllTokenDispatcher"
            assert row["layer"] == 1
            assert row["n_tokens_local"] == 32
            assert [p["phase"] for p in row["intervals"]] == [
                "route",
                "pre",
                "dispatch",
                "compute",
                "combine",
                "post",
            ]
            assert row["n_tokens_recv"] == 32 * 2  # top-2, EP=1: every routed copy stays local
