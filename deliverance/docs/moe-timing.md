# MoE Phase Timing

Per-phase timing of the MoE layer for both the stock `MoEAlltoAllTokenDispatcher` path and
`MoEDecomposedAlltoAllDispatcher`, so the two can be compared directly and overlap can be
quantified. Design notes for the dispatcher itself: [decomposed-a2a-notes.md](decomposed-a2a-notes.md).

## Decisions

| Decision | Status |
| --- | --- |
| Mechanism: CUDA events recorded on the stream doing the work; resolved off the hot path | Agreed |
| Log every layer of every step (no sampling) | Agreed |
| Do not log `D` or the schedule (possible future extension) | Agreed |
| Timer module lives in `megatron/core/transformer/moe/` (e.g. `moe_timing.py`) | Agreed |
| Offline analysis script: `deliverance/scripts/analyse_moe_timing.py` | Done |
| Off by default; enabled by env var `MOE_PHASE_TIMING_DIR` (set = on, value = per-rank output dir) | Agreed |
| No inference-engine plumbing. The timer derives step boundaries itself and buckets by token count instead of step type | Agreed |

## Usage

All paths are relative to `<MEGATRON_ROOT>`. Run every command from there with the venv active:

```bash
cd <MEGATRON_ROOT>
source .venv/bin/activate
```

### Enable timing on a run

Set `MOE_PHASE_TIMING_DIR` to a **fresh** directory. `torch.distributed.run` passes it to every
rank, and each rank writes `rank<NNNNN>.jsonl` there. The launcher's `--timing-dir` sets it for you
(and refuses a directory that already has rank files):

```bash
bash deliverance/scripts/run_moe_inference.sh \
  --model mistralai/Mixtral-8x7B-v0.1 --dataset gsm8k --samples 2 --max-new-tokens 8 \
  --timing-dir local/moe-timing/$(date +%Y%m%d-%H%M%S)
```

Leave the variable unset to run without timing (zero overhead apart from one env lookup). If the
directory already contains a `rank<NNNNN>.jsonl`, the run stops with `FileExistsError` instead of
overwriting it.

Verified end-to-end on 2026-10-27: Mixtral 8×7B, legacy TP=4 checkpoint, `--parallelism dp`
(TP=4, DP=2, EP=1), 4 gsm8k prompts, 8 new tokens. All 8 ranks wrote 608 rows (19 steps × 32
layers), every row with all six phases. Generated text was identical to earlier runs without
the timer.

```bash
# The run above (TP=4, DP=2, EP=1), with the current launcher:
bash deliverance/scripts/run_moe_inference.sh \
  --model mixtral-8x7b-tp4-pp1-ep1 --tp 4 --ep 1 --dataset gsm8k --samples 4 --max-new-tokens 8 \
  --timing-dir local/moe-timing/$(date +%Y%m%d-%H%M%S) \
  --tokenizer-type Llama2Tokenizer \
  --tokenizer-model /depot/shared/.cache/huggingface/hub/models--mistralai--Mixtral-8x7B-v0.1/snapshots/fc7ac94680e38d7348cfa806e51218e6273104b0/tokenizer.model
```

Also verified on 2026-10-27 at the default TP=1, PP=1, EP=8 (`--model mistralai/Mixtral-8x7B-v0.1`,
same prompts): all 8 ranks wrote 608 rows with all six phases on `MoEAlltoAllTokenDispatcher`, and
`n_tokens_recv` varied per rank as expected under EP. `--timing-dir` sets `MOE_PHASE_TIMING_DIR`.

### Analyse a run

```bash
python deliverance/scripts/analyse_moe_timing.py local/moe-timing/<run>                  # drops steps < 5
python deliverance/scripts/analyse_moe_timing.py local/moe-timing/<run> --warmup 0 --per-layer
python deliverance/scripts/analyse_moe_timing.py local/moe-timing/<run> --csv local/<run>.csv
```

It prints prefill and decode tables (see [Metrics](#metrics-offline)): makespan and
exposed comm / exposed compute / overlapped / other for the critical rank (mean, p50, p99, max,
% of makespan), the mean exposed comm over ranks, and per-phase sums. `--per-layer` adds mean
makespan per layer, which shows where cold start sits. `--csv` writes one row per (step, layer).

- **Step kind:** a step is decode if every rank's `n_tokens_local` ≤ `--decode-max-tokens`
  (default 64, the padded decode size seen so far), else prefill.
- **Warmup vs prefill:** in an offline run every prompt arrives at t = 0, so step 0 is the *only*
  prefill step, and its first layers carry the cold start (Mixtral EP=8: layer 1 of step 0 took
  1.3 s vs ~32 ms warm). The default `--warmup 5` therefore leaves no prefill samples, and
  `--warmup 0` keeps prefill with the cold start included (use p50 or `--per-layer`). Doing a
  warmup `generate` inside `offline_inference.py` would remove this trade-off. Not done yet.

### Smoke test (unit tests)

Timer tests on one GPU (about 1 s):

```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc-per-node 1 \
  -m pytest -q -p no:cacheprovider tests/unit_tests/transformer/moe/test_moe_timing.py
```

Expected: `6 passed`.

Regression check against the existing MoE layer tests (timing off by default, so outcomes must
match a run without the hooks):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc-per-node 4 \
  -m pytest -q -p no:cacheprovider \
  tests/unit_tests/transformer/moe/test_moe_layer.py \
  tests/unit_tests/transformer/moe/test_moe_timing.py
```

Expected on 4 GPUs: `20 failed, 49 passed, 8 skipped`. All 20 failures are
`world_size (4) is not divisible by ... (8)`. Those cases need `--nproc-per-node 8`.

### Formatting

isort/black are not in the venv. Run them ephemerally at the versions pinned in `pyproject.toml`:

```bash
uvx isort==5.13.2 <files>
uvx black==26.3.0 <files>
```

## Mechanism

- `torch.cuda.Event(enable_timing=True)` records a GPU timestamp when the stream reaches it.
  All events on one device share a clock, so every event is expressed as an offset from a
  per-layer reference event. That gives a full per-rank timeline with **no host sync in the hot path**.
- Megatron's `megatron/core/timers.py` is **not** used: it synchronizes the device, which would
  destroy the overlap being measured.
- Events come from a pool to avoid allocation churn.
- Resolution is lazy and non-blocking: at the start of a new step, resolve earlier steps whose
  last event reports `query() == True`, and keep the rest pending. A blocking final flush runs at
  shutdown.
- NVTX ranges (`moe.<phase>` / `moe.<phase>[idx]`) are emitted alongside, so Nsight Systems traces
  can validate the event numbers. Nsight is for validation only, not the primary metric. They go
  through Megatron's `nvtx_range_push/pop`, so they appear only when Megatron's NVTX profiling is
  enabled (`configure_nvtx_profiling(True)`). NVTX ranges mark CPU issue time; Nsight maps them to
  the GPU kernels.
- The layer reference event is recorded at the top of `MoELayer.forward`, before the shared
  experts and the router, so `T` covers the whole layer. Timing is skipped for partial CUDA-graph
  passes (`intermediate_tensors`), during graph capture, and under MoE-layer recompute.
- Output files are opened with mode `x`: an existing `rank<NNNNN>.jsonl` is never overwritten, and
  the run fails fast with a message to use a fresh directory.

## What is recorded

### Stock path (sequential on the main stream)

Event pairs around the existing `MoELayer` methods:

| Phase | Method |
| --- | --- |
| `route` | `route` (router) |
| `pre` | `preprocess` (permute + metadata, incl. DtoH sync) |
| `dispatch` | `dispatch` (EP A2A) |
| `compute` | `routed_experts_compute` (`dispatch_postprocess` + `experts()` + `combine_preprocess`) |
| `combine` | `combine` (EP A2A) |
| `post` | `postprocess` (unpermute) |

### Decomposed path (three streams)

| Phase | Stream | Granularity |
| --- | --- | --- |
| `pre` | main | once, as stock |
| `schedule_cpu` | CPU (`perf_counter`) | `decompose` + `order` + batching |
| `dispatch[k]` | `s_disp` | per matching k |
| `compute[b]` | `s_comp` | per coalesced batch b |
| `combine[k]` | `s_comb` | per matching k |
| `post` | main | once, as stock |

### Record schema (one JSONL row per rank × step × layer)

```text
{ rank, step, layer, path,                  # path = dispatcher class name, e.g. "MoEAlltoAllTokenDispatcher"
  n_tokens_local, n_tokens_recv,             # bucket on these instead of step type
  schedule_cpu_us,                           # decomposed only
  intervals: [ { phase, idx, tokens, start_us, end_us } ... ]   # relative to layer reference event
}
```

`idx` is `k` for dispatch/combine, `b` for compute, and null for stock phases. `tokens` is the
token count moved or computed in that interval.

`step` is a counter local to the timer: it increments each time the first MoE layer
(lowest `layer_number`) starts a forward pass. `layer` is `MoELayer.layer_number`. Neither needs
information from the inference engine.

## Metrics (offline)

Per rank, per (step, layer), on that rank's clock (intervals on all streams share it):

- `W = [start(route), end(post)]`, so `|W|` is the rank's layer makespan.
- `C` = union of `compute` intervals (expert compute only).
- `M` = union of `dispatch` and `combine` intervals.

| Metric | Definition | Purpose |
| --- | --- | --- |
| Makespan | `\|W\|` | Primary metric, head-to-head vs stock |
| Exposed comm | `\|M \ C\|` | Communication with no expert compute under it: the paper's "bubbles" |
| Exposed compute | `\|C \ M\|` | Compute with no communication under it |
| Overlapped | `\|M ∩ C\|` | Communication hidden under compute |
| Other | `\|W\| − \|M ∪ C\|` | `route`, `pre`, `post`, idle gaps. Not compute |
| `pre`, `post` | their own durations | Side information only |
| Compute cost curve | `compute[b]` duration vs `tokens` | Measured Fig. 1: fixed cost `c`, the knee, and choice of τ (later) |

The four middle rows sum to `|W|`. For the stock path, overlapped = 0 and exposed comm =
`dispatch + combine`.

Across ranks there is **no shared clock** (each rank's offsets are from its own reference event):

- **System makespan** for a (step, layer) = **max over ranks** of `|W|`. The layer's A2A re-syncs
  ranks, so this approximates completion time.
- Exposed/overlapped metrics are reported for the **critical rank** (the one with the max `|W|`),
  plus the mean across ranks.

Reporting: drop the first **K = 5** steps (warmup), then split **prefill vs decode** and give
mean / p50 / p99 / max.

> [!IMPORTANT]
> **"Exposed comm" includes waiting, not just transfer.** A `dispatch`/`combine` interval spans
> from the transfer's start event to its end event, which covers time blocked on the slowest peer.
> That delay therefore *inflates the communication time* (and exposed comm), rather than
> disappearing from it. That is the intended accounting for now. Separating wait from transfer is
> a future extension.

<!-- separate callouts -->

> [!IMPORTANT]
> **With TP > 1, `compute` contains TP communication.** The TP all-gather and reduce-scatter run
> inside `routed_experts_compute`, so they count toward `C` (compute) and *reduce* apparent exposed
> comm. Exposed-comm numbers are only clean at TP = 1. The analysis script repeats this on every run.

## Caveats

Observed in the first real run (TP=4, EP=1):

- **With TP>1, TP communication is counted in `compute`.** The stock path's TP all-gather
  (`dispatch_postprocess`) and reduce-scatter (`combine_preprocess`) run inside
  `routed_experts_compute`, so `dispatch`/`combine` only cover the EP all-to-all. At EP=1 they are
  near zero (`combine` = 0 µs). Attribute phases accordingly when TP>1.
- **Warmup:** steps 0 and 1 took 1.2 s and 124 ms per layer vs ~6.5 ms in steady state. The
  analysis drops the first 5 steps by default.
- **Token counts look padded.** Decode steps with 2 live requests per replica reported
  `n_tokens_local = 64`. The engine pads tokens per step; see the padding item in the notes
  checklist. The mechanism isn't confirmed yet.
- **Occasional `pre` spikes** (0.5 ms → 1.5–3 ms on some steps), consistent with the host sync in
  `preprocess`. Not investigated.

- **A transfer's measured duration includes waiting for its partner.** P2P kernels block until
  the partner arrives, so `dispatch[k]` = wait + transfer. Events alone cannot separate them.
  Report it as "stream occupied by transfer".
- **Stock `compute` is not pure experts().** It includes `dispatch_postprocess` and
  `combine_preprocess` (sort/unsort; no-ops with 1 local expert, TP=1). Keep that in mind when
  comparing `B_comp` across paths.
- **Warmup.** NCCL sets up each P2P connection the first time a pair talks, which is very slow.
  Warm up all pairs and drop warmup steps.
- **Bucket by token count.** Prefill and decode differ by orders of magnitude in N. Without
  engine information, `n_tokens_local` is the proxy for step type.
- **Dummy forwards cannot be identified** without engine information. They are logged like any
  other step (typically small or zero `n_tokens_local`). They are real EP participation, so
  including them is defensible. If they distort results, revisit adding engine metadata.
- **Fair baseline.** Disable CUDA graphs for both paths. Events inside captured graphs need
  special handling, and graphs change the stock path's cost.
- **`schedule_cpu` sits between the DtoH sync and the first P2P**, so the GPU is idle then and
  it shows up in `T`.
- **Log volume.** Every layer of every step: steps × 32 layers × 8 ranks rows. Write per-rank files
  and buffer the writes.

## Checklist

- [x] Decide how timing is enabled: env var `MOE_PHASE_TIMING_DIR`.
- [x] `moe_timing.py`: event pool, `moe_phase(record, name, idx, tokens)` context manager,
      interval bookkeeping, lazy resolution, local step counter, per-rank JSONL writer.
- [x] Hooks in the stock `MoELayer` phases.
- [x] Unit tests: `tests/unit_tests/transformer/moe/test_moe_timing.py`.
- [ ] Hooks inside `dispatch_compute_combine` (per matching / per batch) + `schedule_cpu`.
- [x] NVTX ranges with matching tags.
- [ ] Run `test_moe_layer.py` on all 8 GPUs (20 cases need world size 8; only checked on 4 so far).
- [x] Offline analysis script: metrics table (`analyse_moe_timing.py`).
- [ ] Compute cost curve (`compute[b]` duration vs tokens), once the decomposed path exists.
- [ ] Decide on a real warmup in `offline_inference.py` (throw-away `generate`, record warmup step
      count) so prefill can be measured without cold start.
- [ ] Validate against an Nsight Systems trace on a few steps.
- [ ] Overhead check: stock path with timing on vs off.

## Future extensions

- Log `D` and the schedule per layer, to replay traces in the trace-driven simulator and compare
  predicted vs measured makespan.
- Separate P2P wait vs transfer time (e.g. pair-side comparison or Nsight correlation).
- Engine metadata (step index, prefill/decode, dummy flag) passed from the inference engine, if
  token-count bucketing proves insufficient.
