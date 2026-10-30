# Decomposed All-to-All: Implementation Plan

How `MoEDecomposedAlltoAllDispatcher` gets into Megatron and how it is compared with the stock
`MoEAlltoAllTokenDispatcher`. Each step ends with a check that must pass before the next one
starts.

- Design, decisions and pseudocode: [decomposed-a2a-notes.md](decomposed-a2a-notes.md)
- Timing and metrics: [moe-timing.md](moe-timing.md)
- Runs: [moe-inference.md](moe-inference.md)

## Scope

This is an adaptation of the OCS paper, not an emulation of OCS. Only the decomposition of `D`
into matchings is borrowed, to split communication into stages and create overlap. There is no
barrier and no reconfiguration delay. The one change to system behaviour is that the single
batched alltoallv becomes several rounds of P2P sends and receives. `D` is taken as the system
produces it, padded and dummy-forward tokens included.

Supported initially: inference, TP=1, EP>1 with any number of local experts per rank (dev
target: Mixtral 8×7B
at EP=8), no CUDA graphs, dropless routing. Training comes later
(see the notes, "Training (later)").

## Usage

All paths are relative to `<MEGATRON_ROOT>`. Run every command from there with the venv active:

```bash
cd <MEGATRON_ROOT>
source .venv/bin/activate
```

Test files and flags named below are **planned**. The commands work once the step that adds
them is done.

## Overview

| Step | What | Main files |
| --- | --- | --- |
| 1 | Config fields + validation | `megatron/core/transformer/transformer_config.py` |
| 2 | Register the dispatcher type | `megatron/core/transformer/moe/moe_layer.py` |
| 3 | Decomposition + ordering (CPU) | `megatron/core/transformer/moe/decomposition.py` (new) |
| 4 | Dispatcher preprocessing: full `D`, schedule, offsets | `megatron/core/transformer/moe/token_dispatcher.py` |
| 5 | `dispatch_compute_combine()` + `MoELayer.forward` hook | `token_dispatcher.py`, `moe_layer.py` |
| 6 | Correctness tests vs stock | `tests/unit_tests/transformer/moe/` |
| 7 | Timing hooks in the decomposed path | `megatron/core/transformer/moe/moe_timing.py`, `token_dispatcher.py` |
| 8 | End-to-end runs and the comparison | `deliverance/scripts/run_moe_inference.sh`, `analyse_moe_timing.py` |

## Gaps found in the code

Things the design depends on that the current code does not provide. Each is handled in the
step noted.

1. **Only part of `D` reaches the CPU (step 4).** `MoEAlltoAllTokenDispatcher.preprocess()`
   all-gathers `num_global_tokens_per_expert` (`[tp, ep, num_experts]`) over the TP×EP group,
   but keeps it in a local variable. The DtoH copy in `_maybe_dtoh_and_synchronize()` only moves
   this rank's row (`input_splits`) and column (`output_splits`) to the CPU. Every rank needs
   the full `n × n` matrix to compute the same schedule.
2. **Code that tests for exactly `"alltoall"` (step 1).** A new type name silently drops out of
   every branch that checks for the string `"alltoall"`:

   | Location | What it controls |
   | --- | --- |
   | `megatron/core/models/common/utils.py:93` | `free_mlp` (training, delayed-wgrad overlap) |
   | `megatron/core/transformer/transformer_layer.py:528` | pre-MLP layernorm recompute with MoE CUDA graphs |
   | `transformer_config.py:2218` | `moe_shared_expert_overlap` allowed dispatchers |
   | `transformer_config.py:3337` | `moe_preprocess` CUDA graph check |
   | `transformer_config.py:3510` | `overlap_moe_expert_parallel_comm` allowed dispatchers |
   | `transformer_config.py:3736`, `:3838` | other alltoall-only asserts |

   Each needs a deliberate decision. Most are training or CUDA-graph paths that step 1
   rejects anyway, so the usual answer is "leave it, the new type is excluded".
3. **scipy is installed only indirectly (step 3).** It comes in through `nvidia-modelopt` and is
   not declared in `pyproject.toml`. Import it inside `decompose()`, not at module level, so
   only the new path depends on it.

## Step 1: `TransformerConfig` patch

CLI flags are generated from `TransformerConfig` fields by `ArgumentGroupFactory`: a `Literal`
becomes `choices` and the docstring becomes `--help`. **No edit to
`megatron/training/arguments.py` is needed.** `--use-checkpoint-args` does not override
`moe_token_dispatcher_type` when it is given on the command line (`force=False`), and the new
fields are not stored in checkpoints.

### Fields

```python
moe_token_dispatcher_type: Literal['allgather', 'alltoall', 'flex', 'decomposed_alltoall'] = "allgather"

moe_decomposed_coalesce_tokens: Optional[int] = 0
"""Coalescing threshold τ (tokens) for 'decomposed_alltoall'. Received chunks are grouped until
they hold at least τ tokens, then one experts() call runs. 0 = one call per matching;
None = one call after the last matching (same GEMM shape as the stock dispatcher)."""

moe_decomposed_decompose: Literal['greedy_mw'] = 'greedy_mw'
"""Decomposition of the demand matrix into matchings, for 'decomposed_alltoall'."""

moe_decomposed_order: Literal['largest_first', 'as_is'] = 'largest_first'
"""Order in which matchings are issued, for 'decomposed_alltoall'."""
```

Flags: `--moe-token-dispatcher-type decomposed_alltoall`, `--moe-decomposed-coalesce-tokens`,
`--moe-decomposed-decompose`, `--moe-decomposed-order`.

### Validation in `__post_init__` (only when the type is `decomposed_alltoall`)

| Check | Reason |
| --- | --- |
| `expert_model_parallel_size > 1` | no all-to-all at EP=1 |
| `tensor_model_parallel_size == 1`, `expert_tensor_parallel_size in (None, 1)` | TP>1 not supported |
| `cuda_graph_impl == "none"` | the schedule depends on the data |
| `not moe_pad_expert_input_to_capacity`, `moe_expert_capacity_factor is None` | no drop-and-pad or dropping |
| `transformer_impl != "inference_optimized"` | it swaps the dispatcher out during inference |
| `not moe_shared_expert_overlap`, `not overlap_moe_expert_parallel_comm` | other overlap paths own the streams |
| `moe_decomposed_coalesce_tokens is None or >= 0` | τ range |

Then go through the gap-2 locations one by one.

**Status (2026-10-28): done.** Validation is in `TransformerConfig._validate_decomposed_alltoall()`,
called from `__post_init__`. Gap-2 review: no change needed at any location. Each is either a
setting the validation above already rejects (shared-expert overlap, EP comm overlap, CUDA graphs)
or a training-only path (`free_mlp`, batch-invariant training, sequence packing).
`moe_router_padding_for_quantization` is allowed: its padding goes into `routing_map`, so it is
part of `D` like any other routed token. There is no local-expert-count check: more than one
local expert per rank is supported, and Megatron's own `MoELayer` asserts that the experts
divide evenly over EP. Tests: `test_decomposed_config.py`, 16 passed.

### Check

```bash
# Flags parse (prints: decomposed_alltoall 512 greedy_mw as_is).
# `--help` itself is broken upstream (a '%' in another option's help text), unrelated to this patch.
python -c "import sys; sys.argv=['x','--moe-token-dispatcher-type','decomposed_alltoall','--moe-decomposed-coalesce-tokens','512','--moe-decomposed-order','as_is']
from megatron.training.arguments import parse_args; a=parse_args()
print(a.moe_token_dispatcher_type, a.moe_decomposed_coalesce_tokens, a.moe_decomposed_decompose, a.moe_decomposed_order)" 2>/dev/null | tail -1

# Config tests: a valid config builds; each unsupported combination raises
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc-per-node 1 \
  -m pytest -q -p no:cacheprovider tests/unit_tests/transformer/moe/test_decomposed_config.py
```

## Step 2: Register the dispatcher type

- Add a `decomposed_alltoall` branch next to `alltoall` in `MoELayer.__init__`
  (`moe_layer.py:351`).
- In `forward`, assert that the active dispatcher is still the decomposed one after the
  `_inference_token_dispatcher` swap block. Step 1 already blocks `inference_optimized`; this
  is a second line of defence.
- Launcher: no change. `--dispatcher decomposed_alltoall` sets the type, and the
  `moe_decomposed_*` flags go after `--` (passed straight to Megatron).
- Until step 5, the class only inherits stock behaviour, so a run must give stock results.

### Check

```bash
# Stock behaviour under the new type name (subclass not yet overriding anything)
bash deliverance/scripts/run_moe_inference.sh \
  --model mistralai/Mixtral-8x7B-v0.1 --dataset gsm8k --samples 4 --max-new-tokens 8 \
  --dispatcher decomposed_alltoall
```

Expected: same generated text as `--dispatcher alltoall`. The timing `path` field reads
`MoEDecomposedAlltoAllDispatcher` when `--timing-dir` is set.

## Step 3: Decomposition and ordering (CPU, deterministic)

New module `megatron/core/transformer/moe/decomposition.py`, with no torch.distributed and no GPU:

- `decompose(W) -> list[perm]`: greedy max-weight (Jonker-Volgenant, `linear_sum_assignment`) on
  the complete bipartite graph of rank pairs, minus the diagonal. Each round takes the
  max-weight perfect matching among the pairs not yet used, counting zero-weight pairs, and then
  removes those pairs from the graph (see the notes' pseudocode).
  - What remains after each round is regular, so a perfect matching always exists. The result
    is **exactly `n − 1` matchings** (`n` with the diagonal), with each off-diagonal pair in
    exactly one matching.
  - The schedule is exactly what this returns; only `order()` changes it afterwards.
- **Diagonal not routed.** `D[me, me]` (local tokens) is in no matching and never goes through
  P2P. It still goes through `experts()`. It is chunk 0 of the receive buffer: one on-device
  copy, as the stock alltoallv's self-send does, so the τ coalescing stays one slice. It is
  computed first, while matching 1 is in flight (step 5).
- `order(matchings, D)`: `largest_first` (stable sort by the largest entry) or `as_is`.
- **Zero-token pairs are normal.** Every pair is in some matching, so a pair with `D = 0` still
  appears. With sparse `D`, whole matchings can carry zero tokens. In random tests at 30%
  density that was about one in three matchings at `n = 8`. Within one matching a rank can
  have nothing to send and still have something to receive. Step 5 posts each P2P op according
  to its own count, so this needs no change to the schedule.
- Deterministic tie-breaking: identical `D` gives an identical schedule on every rank.
- Inputs are CPU numpy arrays and nothing is moved to the GPU, so no host sync.

### Check

Unit tests (`test_decomposition.py`):

- the matchings cover `D` exactly (each nonzero entry in exactly one matching)
- exactly `n − 1` matchings, each a permutation with no fixed points
- every off-diagonal pair appears in exactly one matching
- the output is exactly what the greedy loop produces (no post-processing besides `order()`)
- the same input gives the same output across repeated calls
- a small hand-checked example (paper Fig. 2)
- edge cases: all-zero `D`, diagonal-only `D`, one dense row

```bash
python -m pytest -q -p no:cacheprovider tests/unit_tests/transformer/moe/test_decomposition.py
```

## Step 4: Dispatcher preprocessing

In `MoEDecomposedAlltoAllDispatcher`:

- Override `preprocess()`: call the parent method, then keep the full count tensor. At TP=1,
  `G = counts.view(ep, ep, E)` with `G[src, dst, e]` (E = local experts per rank),
  `D = G.sum(-1)` and `cnt_e = G[:, me, :]` (tokens from each source to each of my experts).
- Add `G` to the existing DtoH copy, on the same stream and before the same event, so it costs
  no new sync. `cnt_e` must come from `G`: the stock path does not copy its own per-expert
  counts to the CPU when `moe_permute_fusion` is on.
- Override `dispatch_preprocess()`: call the parent method (stock permute), then, after the
  existing sync, build `schedule`, `send_off`, `src_k`, `cnt_k`, `recv_off` and the τ batches as
  in the pseudocode.
- No `.item()` or `.tolist()` on GPU tensors anywhere after this point.

### Check

Multi-GPU test on random routing at EP=8:

- `D` is identical on all ranks (all-gather it and compare)
- `D[me, :] == input_splits` and `D[:, me] == output_splits`
- with E > 1: `cnt_e.sum(0)` equals the stock path's tokens per local expert
- `schedule` is identical on all ranks

```bash
python -m torch.distributed.run --nproc-per-node 8 \
  -m pytest -q -p no:cacheprovider tests/unit_tests/transformer/moe/test_decomposed_dispatcher.py -k preprocess
```

## Step 5: `dispatch_compute_combine()` and the forward hook

- `__init__`:
  - two extra communicators over the EP ranks, one for dispatch and one for combine, created
    with `new_group` from `pg_collection.ep`'s ranks
  - eager init of both communicators (a tiny all-reduce on each). Every rank must take part in
    the first call on a new communicator, and the zero-count rule means not every rank joins
    the first matching.
  - three streams: `s_disp`, `s_comp`, `s_comb`
- `dispatch_compute_combine(x, p, experts)`: the three loops from the pseudocode.
  - dispatch: one `batch_isend_irecv` per matching (one send and one receive). Never a separate
    `isend` + `irecv`: the group is what rules out the cycle deadlock (notes, System flags).
    Then `work.wait()` on the returned works with `s_disp` current (GPU-side wait), then record
    `ev_disp[k]`. Combine does the same on `s_comb`.
  - when both of this rank's counts in a matching are 0, make no call: `batch_isend_irecv([])`
    raises.
  - **Post each op according to its own count.** A send is posted only if `D[me, dst] > 0`, and
    a receive only if `D[src, me] > 0`, decided separately. Every rank holds the same `D`, so
    both ends of a pair always agree on whether the op exists, and no rank waits on a peer that
    never posts. Never skip a whole matching because this rank has nothing to send. If neither
    op exists, the matching issues nothing on this rank. The same rule applies to combine,
    with the directions reversed.
  - `perm[me] != me` always, because the diagonal is excluded.
  - local chunk: copied into receive chunk 0 on `s_comp` at the start, so the compute stream
    can start without waiting on dispatch. After its batch is computed, `s_comb` copies it back
    to `out[send_off[me]]` before the first combine P2P. This copy, compute, copy sequence is
    what the stock alltoallv does for the self-chunk.
  - compute: one `experts()` call per τ batch, after waiting on the batch's last `ev_disp`, with
    `tokens_per_expert` summed from `cnt_e` over the batch's sources (CPU). A batch with 0 tokens
    skips `experts()` (nothing to compute, move on) but still records `ev_comp[b]`, so combine's
    waits stay uniform.
  - multiple local experts: a single-source batch is already expert-contiguous (stock
    permutation 1), so it needs no sort. A batch spanning several sources with E > 1 is
    expert-sorted before `experts()` and un-sorted after. Reuse the stock `sort_chunks_by_idxs`
    on the batch slice, with that batch's per-(source, expert) counts. The communication
    itself never changes.
  - combine: inverse matching, same order as dispatch
- The receive buffers are in schedule order. `out` is in the stock layout, so the stock
  `combine_postprocess` is unchanged.
- `record_stream` on every buffer used across streams. The main stream waits on `s_comb` at the
  end.
- Ranks never wait on anything except their P2P partners.
- `MoELayer.forward`: if the dispatcher is `MoEDecomposedAlltoAllDispatcher`, replace
  `dispatch` → `routed_experts_compute` → `combine` with one call. Pass `self.experts` so the
  dispatcher runs the same expert module.

### Check

- With τ=None, run step 6's bitwise test.
- Run 100 iterations without a hang at τ=0 and τ=None.

## Step 6: Correctness tests

`tests/unit_tests/transformer/moe/test_decomposed_dispatcher.py`, modelled on
`test_a2a_token_dispatcher.py`. Same inputs and weights through the stock and the decomposed
dispatchers at EP=8:

| Case | Expectation |
| --- | --- |
| τ=None (one call, same GEMM shape as stock) | bitwise equal |
| τ=0, middle τ | `allclose` (different GEMM sizes can choose different kernels) |
| skewed `D`, sparse `D` (empty pairs), all tokens to one expert | as above |
| zero tokens on some ranks | no hang, correct output |
| a rank sends nothing in a matching but receives (and the reverse) | no hang, correct output |
| all-zero matchings (sparse `D`) at τ=0 | `experts()` skipped for them, correct output |
| only local tokens (`D` diagonal), or none | no P2P posted, correct output |
| E = 2 and E = 4 local experts (16 and 32 experts at EP=8), τ = None / 0 / middle | as the τ rows above; τ=None sorts the whole buffer once, like stock |
| E > 1, a local expert receives 0 tokens | correct output |

```bash
python -m torch.distributed.run --nproc-per-node 8 \
  -m pytest -q -p no:cacheprovider tests/unit_tests/transformer/moe/test_decomposed_dispatcher.py

# Regression: existing MoE tests on all 8 GPUs (outstanding from moe-timing.md)
python -m torch.distributed.run --nproc-per-node 8 \
  -m pytest -q -p no:cacheprovider \
  tests/unit_tests/transformer/moe/test_moe_layer.py \
  tests/unit_tests/transformer/moe/test_a2a_token_dispatcher.py
```

## Step 7: Timing hooks in the decomposed path

- `schedule_cpu` (`perf_counter` around decompose + order + batching).
- `dispatch[k]`, `compute[b]` and `combine[k]` on their own streams, with `idx` and `tokens`
  filled in.
- Record each start event **after** that stream's `wait()` (notes, OBSERVE LATER). Otherwise
  the interval includes idle time and hides exposed comm.
- P2P runs on PyTorch's internal NCCL stream. A `dispatch[k]` start event on `s_disp` marks when
  the call was issued; the end event (after `work.wait()`) marks completion. Check the gap
  against Nsight.
- `n_tokens_recv` = the sum of `cnt_k`.

### Check

```bash
bash deliverance/scripts/run_moe_inference.sh \
  --model mistralai/Mixtral-8x7B-v0.1 --dataset gsm8k --samples 4 --max-new-tokens 8 \
  --dispatcher decomposed_alltoall \
  --timing-dir local/moe-timing/$(date +%Y%m%d-%H%M%S)
python deliverance/scripts/analyse_moe_timing.py local/moe-timing/<run> --warmup 0 --per-layer
```

Expected: every row has `schedule_cpu_us` and per-matching intervals, and overlapped > 0.
Then confirm on one layer with Nsight Systems that the three streams actually overlap.

## Step 8: End-to-end runs and the comparison

Same settings for both paths (engine prefill settings, CUDA graphs off, same checkpoint), several
trials each.

1. **Output check:** compare generated token IDs, stock vs decomposed. If they diverge, normalise
   end-to-end time per generated token.
2. **Prefill only:** `--max-new-tokens 1`.
3. **Prefill + decode:** natural EOS, with the token cap (see the notes on base Mixtral and EOS).
4. **τ sweep:** 0, a few middle values, None. Plot `compute[b]` duration against tokens to find
   the knee.
5. **Order:** `largest_first` vs `as_is`.

```bash
for disp in alltoall decomposed_alltoall; do
  for trial in 1 2 3; do
    bash deliverance/scripts/run_moe_inference.sh \
      --model mistralai/Mixtral-8x7B-v0.1 --dataset gsm8k --samples 64 --max-new-tokens 1 \
      --dispatcher "$disp" \
      --timing-dir "local/moe-timing/prefill-$disp-$trial" \
      --output "local/moe-inference-results/prefill-$disp-$trial.json"
  done
done

# A τ setting is passed through to Megatron after --
bash deliverance/scripts/run_moe_inference.sh \
  --model mistralai/Mixtral-8x7B-v0.1 --dataset gsm8k --samples 64 --max-new-tokens 1 \
  --dispatcher decomposed_alltoall --timing-dir local/moe-timing/prefill-tau512 \
  -- --moe-decomposed-coalesce-tokens 512
```

Metrics, in priority order: layer makespan, overlap, end-to-end time, TTFT. Comparisons are
unpaired, so compare distributions bucketed by token count, not single runs.

## Later

- Training: a custom `autograd.Function` that replays the schedule in reverse (notes,
  "Training (later)").
- Optimisation backlog in the notes: Johnson's rule ordering, local chunk without copies,
  send and receive decoupled across matchings, merging small
  matchings, adaptive fallback for small decode steps, topology-weighted `W`.
- TP > 1.
- An end-to-end run with E > 1 (e.g. Mixtral at EP=4, which needs its own checkpoint conversion,
  or a model with more experts).
