# Decomposed All-to-All: Design Notes, Optimisation Points, Flags

Working notes for `MoEDecomposedAlltoAllDispatcher`
(`megatron/core/transformer/moe/token_dispatcher.py`). Reference: [ref.pdf](ref.pdf).

> [!IMPORTANT]
> **More than one local expert per rank (`num_local_experts > 1`) has severe implications.**
> Mixtral at EP=8 has 1 expert/rank and hides all of this. DeepSeek, Qwen3, or Mixtral at EP<8 do not.
>
> 1. **Coalescing breaks expert contiguity.** A chunk from one source is already expert-sorted
>    (`[e0 | e1]`), but a coalesced batch from two sources is `[src_a: e0, e1 | src_b: e0, e1]`.
>    Grouped GEMM needs each expert's tokens contiguous, so every batch needs a re-sort before
>    the experts and an un-sort after: an extra permute + copy per batch.
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
| Decomposition: greedy max-weight (JV / `linear_sum_assignment`), whole entries removed per round, ≤ `ep_size` matchings | Draft |
| Expert compute granularity: option B, coalesce received chunks until ≥ τ tokens, then one `experts()` call. τ=0 → one call per matching; τ=∞ → one call at the end (baseline) | Agreed |
| Receive buffer in schedule (arrival) order so consecutive matchings are contiguous and coalescing is a free slice; `out` stays in stock layout | Agreed |
| No global barrier between matchings by default; barrier available as a flag | Proposed |
| Dev target: TP=1, EP=8, one expert per rank | Proposed |

## Checklist

### Before implementation

- [x] Explain the per-chunk `experts()` trade-off (one grouped-GEMM launch per matching vs. the
      Fig. 1 fixed overhead) so the user can decide on chunk granularity. Decided: option B (τ).
- [x] Confirm the schedule-order receive buffer, then update the pseudocode (τ coalescing +
      schedule-order `recv_off`).
- [ ] Register the new dispatcher type (`--moe-token-dispatcher-type`) and route `MoELayer` to it.

### Launcher / run setup (later)

- [ ] Run with EP=8: either add an EP option to `run_moe_inference.sh` (currently hard-codes
      `--expert-model-parallel-size 1`) or drive it through environment variables. Undecided.
- [ ] Select the new dispatcher type from the launcher / env.
- [ ] Get an EP-reshardable Mixtral 8×7B checkpoint (`torch_dist`). Legacy shards cannot
      be resharded to EP=8.

### Inference-engine interaction (later)

- [ ] Padding / idle-rank tokens: the engine MAX-syncs batch dims across EP and runs
      `dummy_forward()` on idle ranks. Those tokens are routed and inflate `D`. Decide whether
      to mask them out of `D` and/or skip sending them.

### Implementation

- [ ] Decomposition + ordering function (CPU, deterministic).
- [ ] `dispatch_compute_combine()` with dispatch / compute / combine streams.
- [ ] Separate NCCL communicators for dispatch and combine.
- [ ] `MoELayer.forward` hook.
- [ ] `--wait-for-slowest` barrier flag (+ optional `--reconfig-delay-us`).

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
- **Schedule must be identical on every rank.** Use deterministic tie-breaking.
- **Host syncs inside the pipeline.** Any `.item()` / `.tolist()` on a GPU tensor inside the
  per-matching loop stalls the CPU and stops further kernel issue. Overlap then silently
  disappears. All split sizes must already be on the CPU (they are, after the existing DtoH sync).
- **Cross-stream memory.** Tensors allocated on one stream and consumed on another need
  `record_stream` (or allocation on the consuming stream), or the caching allocator may reuse
  memory early.
- **`CUDA_DEVICE_MAX_CONNECTIONS`.** Has no effect on Blackwell, so no hardware-queue
  serialization across streams here. It would matter on Hopper/Ampere, where `=1` funnels every
  stream into one queue and CPU issue order then decides the overlap.
- **D is already free.** `preprocess()` all-gathers per-expert counts over the TP×EP group and the
  existing DtoH sync brings them to the CPU.
- **Buffer layout.** Send chunks are contiguous per destination (`input_splits`). Recv chunks
  placed at `output_splits` offsets reproduce the stock layout, so stock `combine_postprocess`
  works unchanged.
- **Not supported initially:** TP>1, CUDA graphs, `drop_and_pad`, `num_local_experts > 1`
  (see the IMPORTANT note at the top).
- **Emulation, not OCS.** Without a barrier, ranks can be in different matchings at once.
  Label such results "ordered P2P", not "OCS".

## Straggler handling (no barrier)

P2P flow control stalls a sender once the receiver's NCCL buffer fills, so progression
self-throttles. With fixed durations, removing the barrier cannot increase makespan. Caveats:

- A slow pair delays everything downstream of it in the dependency chain. It is not hidden.
- SM/HBM/PCIe contention means durations are not actually fixed.
- Not faithful to the paper's model. Keep `--wait-for-slowest` (+ optional `--reconfig-delay-us`).

## Design principle: theory first, testbed second

The design implements the paper's model and is **not** shaped by any particular testbed. It
exposes only generic extension points:

- `decompose(W)` takes a weight matrix. By default `W = D` (tokens).
- `order(matchings)` is pluggable.
- Barrier / reconfiguration delay are flags.

Instance-specific knowledge enters only later, as optimisations through these points (e.g.
passing a bandwidth-weighted `W`). It never changes the structure of the code.

## Pseudocode (inference, fully pipelined)

`n = ep_size`, `me = ep_rank`. A matching is a permutation `perm` of the EP ranks. In matching
`k`, rank `i` sends to `perm[i]` and receives from `perm.inv[i]`.

### Decomposition (CPU, identical on every EP rank)

```text
decompose(W):                              # W: [n, n] weights; default W = D (tokens)
    R = W.copy(); matchings = []
    while R.any():
        perm = max_weight_perfect_matching(R)        # JV / linear_sum_assignment(maximize=True)
                                                     # deterministic tie-breaking
        matchings.append(perm)
        R[arange(n), perm] = 0                       # take whole entries (paper Fig. 2h-n)
    return matchings                                 # ≤ n matchings

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
      s_disp, s_comp, s_comb = Stream(), Stream(), Stream()
      decompose_fn, order_fn = config-selected       # extension points
      wait_for_slowest, reconfig_delay = config flags
      tau = config                                   # coalescing threshold (tokens):
                                                     #   0 -> one experts() call per matching
                                                     #   inf -> one call at the end (baseline)
      assert num_local_experts == 1                  # see IMPORTANT note (re-sort per batch)

  dispatch_preprocess(h, routing_map, probs):
      x, p = super().dispatch_preprocess(h, routing_map, probs)    # stock permute + metadata
      D = sum over local experts of num_global_tokens_per_expert   # [n, n], D[src, dst]
                                                                   # CPU after existing DtoH sync
      schedule = order_fn(decompose_fn(W = D), D)
      send_off = exclusive_cumsum(input_splits)      # x is contiguous per destination (stock)

      # Receive side in SCHEDULE (arrival) order: chunk k = tokens from perm_k.inv[me].
      # Consecutive matchings are adjacent, so a coalesced batch is one slice. Local to this rank.
      src_k    = [perm.inv[me] for perm in schedule]
      cnt_k    = [output_splits[src_k[k]] for k]
      recv_off = exclusive_cumsum(cnt_k)

      # Coalesce consecutive matchings into compute batches of >= tau tokens (CPU, no sync).
      # Batching is per-rank and may differ across ranks: compute is local, and comm order is unaffected.
      batches = greedy_group(range(len(schedule)), cnt_k, tau)   # list of [k_first .. k_last]
      return x, p

  dispatch_compute_combine(x, p, experts):
      recv_x, recv_p = empty(sum(cnt_k), H), empty(sum(cnt_k))   # schedule-order layout
      y   = empty_like(recv_x)                       # expert outputs, schedule-order layout
      out = empty_like(x)                            # combine result, STOCK layout
      s_disp.wait(main); s_comp.wait(main); s_comb.wait(main)

      # Stage 1: dispatch
      with stream(s_disp):
          for k, perm in enumerate(schedule):
              dst, src = perm[me], src_k[k]
              p2p_batch(dispatch_comm,
                        send x[send_off[dst] : +input_splits[dst]] -> dst,
                        send p[same slice]                         -> dst,
                        recv recv_x[recv_off[k] : +cnt_k[k]]       <- src,
                        recv recv_p[same slice]                    <- src)
              ev_disp[k].record()
              if wait_for_slowest:
                  stream_barrier(dispatch_comm)      # GPU-side (tiny all-reduce), NOT a host barrier
                  delay(reconfig_delay)

      # Stage 2: expert compute, one call per coalesced batch
      with stream(s_comp):
          for b, ks in enumerate(batches):
              s_comp.wait(ev_disp[ks.last])          # s_disp is in-order: last done => all done
              sl  = slice(recv_off[ks.first], recv_off[ks.last] + cnt_k[ks.last])
              tpe = [sum(cnt_k[k] for k in ks)]      # 1 local expert: single segment
              y[sl] = experts(recv_x[sl], tpe, recv_p[sl])
              ev_comp[b].record()

      # Stage 3: combine (inverse matching, same matching order as dispatch)
      with stream(s_comb):
          for k, perm in enumerate(schedule):
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

Invariants:

- No host sync inside the three loops. All splits and `tpe` are already on the CPU.
- Every rank issues matchings in the same order on each communicator.
- With `num_local_experts == 1`, `tpe` is a single count and the second permute step is a no-op.
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
- **Diagonal `D[i,i]`:** compute local tokens as "matching 0" while matching 1 is in flight.
- **Small-matching merge:** fold matchings below the compute knee (~256 tokens, Fig. 1) into a
  neighbour, or fall back to stock A2A for the residual.
- **Intra-matching imbalance (§3.3):** split heavy pairs across consecutive matchings.
- **Adaptive fallback:** stock A2A when total tokens are small (decode steps, Fig. 3).
- **Decomposition cost:** cache by `D` for repeated decode shapes. Overlap it with permute.
- **Buffers:** preallocate recv/out buffers across layers.
- **Shared experts:** overlap with matching-1 dispatch (Mixtral has none; relevant for DeepSeek).

## Timing

Per-phase timing (stock vs decomposed, overlap metrics) is specified in
[moe-timing.md](moe-timing.md).

## Training (later)

Backward of dispatch = combine with the same matchings, and vice versa. Needs a custom
`autograd.Function` that stores the schedule and replays it reversed.
