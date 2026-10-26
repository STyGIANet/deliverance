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
| Offline analysis script lives in `deliverance/scripts/` | Proposed |
| Off by default; enabled by env var `MOE_PHASE_TIMING_DIR` (set = on, value = per-rank output dir) | Agreed |
| No inference-engine plumbing. The timer derives step boundaries itself and buckets by token count instead of step type | Agreed |

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
- NVTX ranges with the same tag names are emitted alongside, so Nsight Systems traces can
  validate the event numbers. Nsight is for validation only, not the primary metric.

## What is recorded

### Stock path (sequential on the main stream)

Event pairs around the existing `MoELayer` methods:

| Phase | Method |
| --- | --- |
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
{ rank, step, layer, path: "stock" | "decomposed",
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

| Metric | Definition | Purpose |
| --- | --- | --- |
| `T` (makespan) | layer reference → end of `post`, main stream | Head-to-head vs stock |
| `B_disp`, `B_comp`, `B_comb` | sum of interval durations per stage | Busy time per stage |
| `S` (serial) | `B_disp + B_comp + B_comb` (+ `pre` + `post`) | Time with no overlap |
| Hidden time | `S − T` | Absolute overlap gain |
| Overlap efficiency | `(S − T) / (S − max(B))` | Share of hideable time actually hidden (1.0 = perfect) |
| Exposed comm | time in the window where `s_comp` is idle (interval union) | The paper's "bubbles" |
| Compute cost curve | `compute[b]` duration vs `tokens` | Measured Fig. 1: fixed cost `c`, the knee, and choice of τ |
| Effective layer time | `max over ranks` of `T` | The next layer's A2A re-syncs ranks |

## Caveats

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
- [ ] `moe_timing.py`: event pool, `mark(tag, stream)`, interval bookkeeping, lazy resolution,
      local step counter, per-rank JSONL writer.
- [ ] Hooks in the stock `MoELayer` phases.
- [ ] Hooks inside `dispatch_compute_combine` (per matching / per batch) + `schedule_cpu`.
- [ ] NVTX ranges with matching tags.
- [ ] Offline analysis script: metrics table + compute cost curve.
- [ ] Validate against an Nsight Systems trace on a few steps.
- [ ] Overhead check: stock path with timing on vs off.

## Future extensions

- Log `D` and the schedule per layer, to replay traces in the trace-driven simulator and compare
  predicted vs measured makespan.
- Separate P2P wait vs transfer time (e.g. pair-side comparison or Nsight correlation).
- Engine metadata (step index, prefill/decode, dummy flag) passed from the inference engine, if
  token-count bucketing proves insufficient.
