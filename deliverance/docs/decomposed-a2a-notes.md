# Decomposed All-to-All: Design Notes, Optimisation Points, Flags

Working notes for `MoEDecomposedAlltoAllDispatcher`
(`megatron/core/transformer/moe/token_dispatcher.py`). Reference: [ref.pdf](ref.pdf).
Implementation plan: [decomposed-a2a-plan.md](decomposed-a2a-plan.md).

> [!IMPORTANT]
> **More than one local expert per rank (`num_local_experts > 1`) is supported, and it costs
> compute.** Mixtral at EP=8 has 1 expert/rank and hides all of this. DeepSeek, Qwen3, or Mixtral
> at EP<8 do not.
>
> Communication is unaffected, **because of the max-weight decomposition.** It realises each
> `(src, dst)` pair in exactly one matching, so each transmission is the full `D[src, dst]`
> chunk. The stock permute has already sorted that chunk by expert. That is exactly what the
> stock alltoallv moves for that pair, so the transport never needs to know which token goes
> to which expert, and no extra work is needed. A decomposition such as BvN can realise one
> pair over several matchings, splitting `D[src, dst]` into pieces. Then either the pieces need
> per-expert bookkeeping (which expert's tokens are in which piece), or, if sent in stock order,
> the later experts' tokens wait for later matchings. Keep this in mind before swapping
> `decompose()`.
>
> The costs are all in compute:
>
> 1. **Coalescing breaks expert contiguity.** A chunk from one source is already expert-sorted
>    (`[e0 | e1]`), but a coalesced batch from two sources is `[src_a: e0, e1 | src_b: e0, e1]`.
>    Grouped GEMM needs each expert's tokens contiguous, so a batch spanning several sources
>    needs a re-sort before the experts and an un-sort after: an extra permute + copy per batch.
>    This is the stock "permutation 2" (`sort_chunks_by_idxs`), applied per batch instead of
>    once. Single-source batches (always at τ=0) need no sort.
> 2. **The knee applies per expert, not per matching.** A batch of `n` tokens splits into
>    `E_local` segments of about `n / E_local`. A matching that looks large at rank level can
>    still give each expert a tiny GEMM.
> 3. **Fixed cost scales with `K × E_local`.** Every call reads all local experts' weights
>    (Mixtral: ~350 MB per expert) and pads each expert segment (256-token TE grouped-tensor
>    segments / FP8 alignment). K calls means `K × E_local` weight reads and padding.
> 4. **Theory gap.** Greedy MW on the rank-level `D` maximises tokens per (matching, rank pair),
>    but compute efficiency depends on tokens per (batch, expert). A decomposition that is good at
>    rank level can still fragment expert compute. This is the same fragmentation the paper
>    attributes to BvN, one level down.

## Decisions

| Decision | Status |
| --- | --- |
| Inference first; training should need little extra work (see Training below) | Agreed |
| Pipeline all three stages: dispatch(k+1) ‖ compute(k) ‖ combine(k-1) | Agreed |
| Test model: Mixtral 8×7B on the existing inference setup (`run_moe_inference.sh`) | Agreed |
| Dedicated compute stream; main stream only joins at the end | Agreed |
| Decomposition: greedy max-weight (JV / `linear_sum_assignment`) over the complete bipartite graph of rank pairs, minus the diagonal. Each round takes the max-weight perfect matching among unused pairs (zero-weight included) and removes them. Exactly `n − 1` matchings; each off-diagonal pair in exactly one. The schedule is what it returns; only `order()` changes it | Agreed |
| Diagonal `D[i,i]` is not routed (no P2P). The local chunk is chunk 0 of the receive buffer (one on-device copy, as the stock alltoallv's self-send does) and is computed first, while matching 1 is in flight | Agreed |
| A compute batch with 0 tokens skips `experts()` and only records its event. Compute-side only; the schedule is unchanged | Agreed |
| Expert compute granularity: option B, coalesce received chunks until ≥ τ tokens, then one `experts()` call. τ=0 → one call per matching; τ=∞ → one call at the end (baseline) | Agreed |
| Receive buffer in schedule (arrival) order so consecutive matchings are contiguous and coalescing is a free slice; `out` stays in stock layout | Agreed |
| Adaptation, not OCS emulation: borrow only the decomposition. No barrier and no reconfiguration delay; each rank starts its next matching as soon as its P2P ops for the current one complete | Agreed |
| Dispatcher type name: `decomposed_alltoall` | Agreed |
| τ config: `None` = ∞ (one `experts()` call at the end) | Agreed |
| Take `D` as the system produces it (no masking of padded / dummy tokens); only the transport changes | Agreed |
| `num_local_experts > 1` supported from the start. Communication is unchanged because max-weight realises each `(src, dst)` pair in exactly one matching: a chunk is all of `src`'s tokens for `dst`, as in the stock alltoallv. Compute expert-sorts batches that span several sources | Agreed |
| Dev target: TP=1, EP=8 on Mixtral (one expert per rank); multi-expert correctness tested at unit level | Proposed |

## Checklist

### Before implementation

- [x] Explain the per-chunk `experts()` trade-off (one grouped-GEMM launch per matching vs. the
      Fig. 1 fixed overhead) so the user can decide on chunk granularity. Decided: option B (τ).
- [x] Confirm the schedule-order receive buffer, then update the pseudocode (τ coalescing +
      schedule-order `recv_off`).
- [x] Register the new dispatcher type (`--moe-token-dispatcher-type`) and route `MoELayer` to it.

### Launcher / run setup (later)

- [x] Run with EP=8: `run_moe_inference.sh` takes `--tp/--pp/--ep` (default 1/1/8) and
      prepares the checkpoint from HF on first use ([moe-inference.md](moe-inference.md)).
- [x] Select the new dispatcher type from the launcher: `--dispatcher` passes
      `--moe-token-dispatcher-type` through; it works once the type is registered.
- [x] Get an EP=8 Mixtral 8×7B checkpoint: converted to legacy TP=1/PP=1/EP=8
      (`/depot/shared/megatron-checkpoints/mixtral-8x7b-v0.1-tp1-pp1-ep8`). Token IDs match the
      TP=4/EP=1 checkpoint. Not reshardable; other layouts are converted separately.
- [ ] Natural EOS with the **base** Mixtral: on gsm8k it emits newline tokens (ID 13) rather than
      ending, so "natural EOS" runs may go to the token cap. Decide between the base model with a
      cap, or `mistralai/Mixtral-8x7B-Instruct-v0.1` (same architecture, one conversion).

### Inference-engine interaction (later)

- [x] Padding / idle-rank tokens: the engine MAX-syncs batch dims across EP and runs
      `dummy_forward()` on idle ranks. Those tokens are routed and inflate `D`. Decided: keep
      them. The stock alltoallv sends them too, so `D` is the same demand the stock path serves.

### Implementation

- [ ] Decomposition + ordering function (CPU, deterministic).
- [ ] `dispatch_compute_combine()` with dispatch / compute / combine streams.
- [ ] Separate NCCL communicators for dispatch and combine.
- [ ] `MoELayer.forward` hook.

### Validation

- [ ] Unit test vs `MoEAlltoAllTokenDispatcher` on random routing (EP=8): bitwise-equal at
      τ=∞ (same GEMM shape as stock). Otherwise allclose, because different batch sizes can make
      cuBLAS/TE pick different kernels and accumulation orders.
- [ ] End-to-end Mixtral 8×7B generation matches stock alltoall outputs.
- [ ] Nsight Systems trace confirms dispatch/compute/combine actually overlap.

### Deferred tunables

- [ ] Cap NCCL SM usage (`NCCL_MAX_NCHANNELS` / communicator `max_ctas`) so comm kernels don't
      starve expert GEMMs.
- [ ] Stream priorities (high priority for comm streams).

## System flags (things that bite)

- **EP must be > 1.** Current inference runs use EP=1, so there is no all-to-all.
- **Inference dispatcher swap.** With `transformer_impl == "inference_optimized"`,
  `MoELayer._setup_inference_mode` replaces the dispatcher while `InferenceMode.is_active()`.
  The launcher does not currently set this, but guard against it.
- **CUDA graphs.** The dynamic engine can capture CUDA graphs (including EP dummy steps).
  The schedule is data-dependent, so disable CUDA graphs for this dispatcher.
- **EP batch sync and padding.** With TP=1, EP=8, attention runs 8 DP replicas. The engine
  MAX-syncs batch dimensions across the EP group and runs `dummy_forward()` on idle ranks
  (`dynamic_engine.py`, `batch_dimensions_utils.py`). See the checklist.
- **MoELayer interface.** `dispatch()` → `routed_experts_compute()` → `combine()` are separate
  blocking calls, so overlap needs a fused `dispatch_compute_combine()` hook in `MoELayer.forward`.
- **NCCL op ordering / deadlock.** Ops on a communicator must be issued in the same order on all
  ranks. Dispatch and combine on different streams sharing a communicator can hang, so use
  separate communicators.
- **A matching's send and receive must be one NCCL group** (`batch_isend_irecv`, which wraps the
  `isend` and `irecv` in `ncclGroupStart/End`). In a matching, rank `i` sends to `perm[i]` but
  receives from `perm⁻¹[i]`, usually a different rank: a matching is a permutation made of
  cycles. With send-then-receive on the cycle 0 → 1 → 2 → 0, every rank spins in its send
  waiting for a receive that is never posted, so all three hang. A separate `isend` + `irecv`
  is no safer: each is its own NCCL kernel, and whether this rank's receive can run while its
  own send is still spinning depends on PyTorch's internal stream/communicator choice. A group
  progresses both by contract. NCCL's blocking (a spinning kernel, no backoff or retry, abort
  on the watchdog timeout) is the intended behaviour. Matchings never interleave on a rank:
  groups on one communicator run in issue order on its NCCL stream.
- **`batch_isend_irecv([])` raises** (it reads `p2p_op_list[0]`). When both of a rank's counts
  in a matching are 0, make no call for that matching (the per-op rule below).
- **Every rank must take part in the first call on a new communicator.** PyTorch:
  "if this API is the first collective call in the group … all ranks of the group must
  participate … otherwise, the behavior is undefined." Because of the zero-count rule, some
  ranks may skip the first matching. So initialise the dispatch and combine communicators
  eagerly in `__init__` (e.g. a tiny all-reduce on each) before any P2P.
- **P2P work runs on PyTorch's internal NCCL stream, not on `s_disp`/`s_comb`.** Call
  `work.wait()` on the returned works with `s_disp` (or `s_comb`) current. That is a GPU-side
  wait and does not block the CPU. Record `ev_disp[k]` after it, otherwise the event does not
  mark the transfer's end. For timing, a start event on `s_disp` marks when the call was issued,
  not when NCCL started moving data.
- **Schedule must be identical on every rank.** Use deterministic tie-breaking.
- **Zero-token pairs: post per op, not per rank.** In a matching a rank may send nothing but
  still receive (or the reverse). Post a send only if `D[me, dst] > 0` and a receive only if
  `D[src, me] > 0`. Both ends read the same `D`, so they always agree. Skipping a whole matching
  because there is nothing to send would leave the peer that sends to this rank waiting forever.
- **Host syncs inside the pipeline.** Any `.item()` / `.tolist()` on a GPU tensor inside the
  per-matching loop stalls the CPU and stops further kernel issue. Overlap then silently
  disappears. All split sizes must already be on the CPU (they are, after the existing DtoH sync).
- **Cross-stream memory.** Tensors allocated on one stream and consumed on another need
  `record_stream` (or allocation on the consuming stream), or the caching allocator may reuse
  memory early.
- **`CUDA_DEVICE_MAX_CONNECTIONS`.** Has no effect on Blackwell, so no hardware-queue
  serialization across streams here. It would matter on Hopper/Ampere, where `=1` funnels every
  stream into one queue and CPU issue order then decides the overlap.
- **D is almost free.** `preprocess()` all-gathers per-expert counts over the TP×EP group
  (`num_global_tokens_per_expert`), but that tensor is a local variable. The existing DtoH copy
  only brings this rank's row (`input_splits`) and column (`output_splits`) to the CPU. The
  subclass must keep the full tensor and add it to the same DtoH copy (no extra sync).
- **Buffer layout.** Send chunks are contiguous per destination (`input_splits`). Recv chunks
  placed at `output_splits` offsets reproduce the stock layout, so stock `combine_postprocess`
  works unchanged.
- **Not supported initially:** TP>1, CUDA graphs, `drop_and_pad`. `num_local_experts > 1` is
  supported (see the IMPORTANT note at the top).
- **Adaptation, not OCS.** Only the decomposition is borrowed. There is no barrier and no
  reconfiguration delay, so ranks can be in different matchings at once. Label results
  "ordered P2P", not "OCS".

## Straggler handling (no barrier, by design)

Each matching's P2P on a rank completes once both its send partner and its receive partner have
posted theirs, so ranks progress pairwise and self-throttle. With fixed durations, the absence of
a barrier cannot increase makespan. Caveats:

- A slow pair delays everything downstream of it in the dependency chain. It is not hidden.
- SM/HBM/PCIe contention means durations are not actually fixed.

## Design principle: theory first, testbed second

The design implements the paper's model and is **not** shaped by any particular testbed. It
exposes only generic extension points:

- `decompose(W)` takes a weight matrix. By default `W = D` (tokens).
- `order(matchings)` is pluggable.

Instance-specific knowledge enters only later, as optimisations through these points (e.g.
passing a bandwidth-weighted `W`). It never changes the structure of the code.

## Pseudocode (inference, fully pipelined)

`n = ep_size`, `me = ep_rank`. A matching is a permutation `perm` of the EP ranks. In matching
`k`, rank `i` sends to `perm[i]` and receives from `perm.inv[i]`.

### Decomposition (CPU, identical on every EP rank)

```text
decompose(W):                              # W: [n, n] weights; default W = D (tokens)
    available = ones(n, n, bool); fill_diagonal(available, False)   # diagonal not routed
    matchings = []
    while available.any():
        # max-weight perfect matching over UNUSED pairs (zero-weight pairs count as edges)
        cost = where(available, -W, max(W) + 1)       # JV / linear_sum_assignment
        perm = linear_sum_assignment(cost).col_ind   # deterministic tie-breaking
        matchings.append(perm)
        available[arange(n), perm] = False           # remove used pairs, whatever their weight
    return matchings                                 # exactly n-1 (residual graph is regular)

order(matchings, D):                        # pluggable; default largest-first
    return sorted(matchings, key=lambda perm: -max_i D[i, perm[i]])   # stable sort
```

### Dispatcher

```text
class MoEDecomposedAlltoAllDispatcher(MoEAlltoAllTokenDispatcher):

  __init__(...):
      super().__init__(...)
      dispatch_comm = new_group(ep_ranks)            # separate communicators per comm stage:
      combine_comm  = new_group(ep_ranks)            #   no cross-stream NCCL ordering deadlock
      all_reduce(tiny, dispatch_comm); all_reduce(tiny, combine_comm)   # eager init: all ranks
                                                     #   must join the first call on a new comm
      s_disp, s_comp, s_comb = Stream(), Stream(), Stream()
      decompose_fn, order_fn = config-selected       # extension points
      tau = config                                   # coalescing threshold (tokens):
                                                     #   0 -> one experts() call per matching
                                                     #   inf -> one call at the end (baseline)
      E = num_local_experts                          # >= 1 (see IMPORTANT note at the top)

  dispatch_preprocess(h, routing_map, probs):
      x, p = super().dispatch_preprocess(h, routing_map, probs)    # stock permute + metadata
      G = num_global_tokens_per_expert.view(n, n, E)  # G[src, dst, e]: CPU, same DtoH copy
      D = G.sum(-1)                                  # [n, n], D[src, dst]
      cnt_e = G[:, me, :]                            # [n, E]: tokens from src to my expert e
                                                     # (the stock path does not DtoH this when
                                                     #  moe_permute_fusion is on; G covers it)
      schedule = order_fn(decompose_fn(W = D), D)
      send_off = exclusive_cumsum(input_splits)      # x is contiguous per destination (stock)

      # Receive side in SCHEDULE (arrival) order: chunk 0 = local tokens (diagonal, not routed),
      # chunk k >= 1 = tokens from perm_k.inv[me] (schedule[k-1]).
      # Consecutive chunks are adjacent, so a coalesced batch is one slice. Local to this rank.
      src_k    = [me] + [perm.inv[me] for perm in schedule]
      cnt_k    = [output_splits[src_k[k]] for k]
      recv_off = exclusive_cumsum(cnt_k)

      # Coalesce consecutive matchings into compute batches of >= tau tokens (CPU, no sync).
      # Batching is per-rank and may differ across ranks: compute is local, and comm order is unaffected.
      batches = greedy_group(range(len(cnt_k)), cnt_k, tau)      # list of [k_first .. k_last]
      return x, p

  dispatch_compute_combine(x, p, experts):
      recv_x, recv_p = empty(sum(cnt_k), H), empty(sum(cnt_k))   # schedule-order layout
      y   = empty_like(recv_x)                       # expert outputs, schedule-order layout
      out = empty_like(x)                            # combine result, STOCK layout
      s_disp.wait(main); s_comp.wait(main); s_comb.wait(main)

      # Stage 1: dispatch
      with stream(s_disp):
          for k, perm in enumerate(schedule, start=1):
              dst, src = perm[me], src_k[k]
              # p2p_batch = batch_isend_irecv: ONE NCCL group, then work.wait() on s_disp.
              # each op posted only if its count > 0 (both ends read the same D);
              # no call at all if both are 0 (empty list raises)
              p2p_batch(dispatch_comm,
                        send x[send_off[dst] : +input_splits[dst]] -> dst,
                        send p[same slice]                         -> dst,
                        recv recv_x[recv_off[k] : +cnt_k[k]]       <- src,
                        recv recv_p[same slice]                    <- src)
              ev_disp[k].record()

      # Stage 2: expert compute, one call per coalesced batch
      with stream(s_comp):
          # chunk 0: local tokens, copied on-device (no P2P); ready without waiting on s_disp
          recv_x[0 : cnt_k[0]] = x[send_off[me] : +input_splits[me]]
          recv_p[0 : cnt_k[0]] = p[same slice]
          for b, ks in enumerate(batches):
              if ks.last > 0:
                  s_comp.wait(ev_disp[ks.last])      # s_disp is in-order: last done => all done
              sl  = slice(recv_off[ks.first], recv_off[ks.last] + cnt_k[ks.last])
              tpe = [sum(cnt_e[src_k[k], e] for k in ks) for e in range(E)]   # CPU
              if sum(tpe) > 0:                       # empty batch: nothing to compute, move on
                  if E > 1 and len(ks) > 1:
                      # batch is [src_a: e0..eE | src_b: e0..eE]: expert-sort it (stock
                      # "permutation 2", sort_chunks_by_idxs on this slice), un-sort after
                      xs, ps = sort_by_expert(recv_x[sl], recv_p[sl], cnt_e[src_k[ks]])
                      y[sl] = unsort_by_expert(experts(xs, tpe, ps), cnt_e[src_k[ks]])
                  else:
                      # one source chunk is already [e0 | e1 | ...] (stock permutation 1)
                      y[sl] = experts(recv_x[sl], tpe, recv_p[sl])
              ev_comp[b].record()                    # recorded even when skipped

      # Stage 3: combine (inverse matching, same matching order as dispatch)
      with stream(s_comb):
          # chunk 0: local result back into the stock layout (no P2P)
          s_comb.wait(ev_comp[batch_of(0)])
          out[send_off[me] : +input_splits[me]] = y[0 : cnt_k[0]]
          for k, perm in enumerate(schedule, start=1):
              dst, src = perm[me], src_k[k]
              s_comb.wait(ev_comp[batch_of(k)])
              p2p_batch(combine_comm,
                        send y[recv_off[k] : +cnt_k[k]]             -> src,
                        recv out[send_off[dst] : +input_splits[dst]] <- dst)

      main.wait(s_comb)
      record_stream(...)                              # all cross-stream buffers
      return out                                      # stock layout -> stock combine_postprocess
```

### MoELayer hook

```text
MoELayer.forward:
    route -> preprocess (dispatch_preprocess)
    if isinstance(token_dispatcher, MoEDecomposedAlltoAllDispatcher):
        out = token_dispatcher.dispatch_compute_combine(h, probs, self.experts)
    else:
        out = combine(routed_experts_compute(dispatch(h, probs)))     # stock path
    postprocess (combine_postprocess)
```

> [!WARNING]
> **OBSERVE LATER — where the timing events are recorded decides what "exposed comm" means.**
> Record each `compute[b]` start event **after** `s_comp.wait(ev_disp[...])`, and each `combine[k]`
> start event **after** `s_comb.wait(ev_comp[...])`. If the start event is recorded before the
> wait, the interval includes the stream idling for its input. That inflates compute (`C`), hides
> exposed comm, and makes the decomposition look better than it is.
> When results come in, check this first if exposed comm looks suspiciously low. Also check one
> layer against an Nsight trace.

Invariants:

- No host sync inside the three loops. All splits and `tpe` are already on the CPU.
- Every rank issues matchings in the same order on each communicator.
- Communication never looks at experts: a chunk is all of `src`'s tokens for this rank, already
  expert-sorted by the stock permute, exactly as in the stock alltoallv.
- A batch with one source chunk is already expert-contiguous, so it needs no sort. A batch that
  spans several sources with `E > 1` is expert-sorted before `experts()` and un-sorted after.
  With `E == 1`, `tpe` is a single count and no sort is ever needed.
- The receive buffers (`recv_x`, `recv_p`, `y`) are in schedule order and never leave this
  method. Only `out` is externally visible, and it is in the stock layout.
- Coalescing only changes compute calls. Dispatch and combine still issue one P2P per matching,
  in the same order on every rank.

## Testbed implications (current machine; notes only, not design inputs)

8× RTX PRO 6000 Blackwell, PCIe only, 2 NUMA sockets (`nvidia-smi topo -m`): pairs (0,1),
(2,3), (4,5), (6,7) share a PCIe switch (PIX); same-socket cross-pair is NODE; cross-socket is
SYS (through UPI).

- Pair bandwidths differ, so tokens are not proportional to time. Max-weight-in-tokens
  matchings are not time-balanced here.
- Matchings are not contention-free: cross-socket pairs in one matching share UPI and host
  bridges.
- NCCL may stage cross-socket P2P through host memory. Check the `NCCL_DEBUG=INFO` transport lines.
- Evaluation: results here are "ordered P2P over PCIe", not OCS. Interpret accordingly.

## Optimisation points (backlog)

- **Matching order:** largest-first vs Johnson's rule (3-machine flow shop, §3.3) vs as-is.
- **[testbed] Topology-aware weights:** pass `W = D / bw(i,j)` into `decompose`. Avoid many
  SYS pairs in one matching.
- **Diagonal `D[i,i]`:** done: local tokens are chunk 0, computed while matching 1 is in flight.
- **Send and receive decoupled across matchings:** today a rank's send and receive in matching
  `k` form one group, so the rank cannot move on until both finish. Issuing sends and receives as
  separate ordered streams (for example per-direction communicators) would let a rank's send of
  matching `k+1` start while its receive of matching `k` is still pending. This needs its own
  deadlock argument; the grouped form is the safe baseline.
- **Local chunk without copies:** the local chunk is contiguous in `x` (sorted by destination),
  so `experts()` could read `x[send_off[me] : +input_splits[me]]` and write straight into
  `out[send_off[me] : ...]`. That saves both copies the stock path also makes. Cost: the chunk
  can no longer join a coalesced batch, so it is always its own `experts()` call, and τ=None no
  longer matches the stock GEMM shape (the bitwise test becomes `allclose`). Explore later.
- **Small-matching merge:** fold matchings below the compute knee (~256 tokens, Fig. 1) into a
  neighbour, or fall back to stock A2A for the residual.
- **Intra-matching imbalance (§3.3):** split heavy pairs across consecutive matchings.
- **Adaptive fallback:** stock A2A when total tokens are small (decode steps, Fig. 3).
- **Decomposition cost:** cache by `D` for repeated decode shapes. Overlap it with permute.
- **Buffers:** preallocate recv/out buffers across layers.
- **Shared experts:** overlap with matching-1 dispatch (Mixtral has none; relevant for DeepSeek).

## Benchmark testbed

Metrics, in priority order:

1. **Layer makespan**: from the MoE phase timer ([moe-timing.md](moe-timing.md)).
2. **Overlap**: from the same timer data.
3. **End-to-end completion time**: wall-clock of the `generate()` call.
4. **TTFT**: per-request, already recorded by the engine and written to the results JSON
   (`"ttft"`). No code change needed.

Serving latency under an arrival process is out of scope.

| Decision | Status |
| --- | --- |
| Offline closed-batch runs with `offline_inference`, no client/server | Agreed |
| Keep the coordinator (`--use-coordinator`); one process per trial. No code changes | Agreed |
| Natural EOS (no forced output length). Compare generated token IDs between stock and decomposed runs; if they diverge, normalise end-to-end time per generated token | Agreed |
| Two regimes: prefill-only (`--max-new-tokens 1`) and prefill + decode | Agreed |

Consequences and caveats:

- **Warmup is trimmed in analysis.** Drop the first K steps from the timer data before computing
  layer makespan and overlap.
- **End-to-end time includes first-time costs** (NCCL connection setup, allocator growth, kernel
  compilation). The decomposed path's extra communicators make this cost larger for it. To check:
  `NCCL_RUNTIME_CONNECT=0` (connect all peers at communicator creation) may move NCCL setup out
  of the timed region, depending on the NCCL version.
- **Comparisons are unpaired.** The coordinator assigns prompts to ranks dynamically, so `D`
  differs between runs. Compare distributions (makespan bucketed by token count) over several
  trials, not run against run.
- **Prefill-only regime:** with one output token, end-to-end time ≈ the batch's prefill time
  (≈ max TTFT), with no decode or EOS effects.
- **TTFT here is not serving TTFT.** All requests arrive at t=0, so TTFT = queueing behind other
  prefills + own prefill. It reflects engine prefill scheduling (max tokens per step, chunked
  prefill) as much as the dispatcher, so those settings must be identical across compared runs.
  Report it as per-request prefill completion time.

### Testbed checklist

- [x] Confirm `ttft` is non-null in a coordinator-mode results JSON (first real run). Confirmed
      2026-10-27; `latency` is also populated in coordinator mode.
- [ ] Choose warmup trim K. First run: steps 0–1 are warmup outliers, so K ≥ 2. Confirm on a
      longer run.
- [ ] Check `NCCL_RUNTIME_CONNECT=0` support in the installed NCCL.
- [ ] Generated-token-ID comparison between stock and decomposed runs (analysis step).

## Timing

Per-phase timing (stock vs decomposed, exposed comm/compute) is specified in
[moe-timing.md](moe-timing.md). Before reading results, remember:

- **Exposed comm includes waiting on the slowest peer** (it inflates comm time). See moe-timing.md.
- **TP > 1 puts TP communication inside `compute`.** Exposed comm is only clean at TP = 1.
- **Event placement in the decomposed dispatcher:** see the OBSERVE LATER warning above the
  pseudocode invariants.

## Training (later)

Backward of dispatch = combine with the same matchings, and vice versa. Needs a custom
`autograd.Function` that stores the schedule and replays it reversed.
