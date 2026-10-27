#!/usr/bin/env python3
"""Summarise one MoE phase-timing run (see deliverance/docs/moe-timing.md).

Per rank and (step, layer), on that rank's clock:
  W = [start(route), end(post)]           layer makespan |W|
  C = union of compute intervals          M = union of dispatch + combine intervals
  exposed comm = |M \\ C|, exposed compute = |C \\ M|, overlapped = |M & C|,
  other = |W| - |M | C|                   (the four sum to |W|)

Across ranks there is no shared clock: system makespan is the max over ranks, and
the other metrics are taken from that critical rank (plus the mean over ranks for
exposed comm).

Usage:
  python deliverance/scripts/analyse_moe_timing.py local/moe-timing/<run> [--warmup 5] [--csv out.csv] [--per-layer]
"""

import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

COMM = ("dispatch", "combine")
COMPUTE = ("compute",)


def union(intervals):
    """Merge (start, end) pairs into disjoint sorted intervals."""
    merged = []
    for s, e in sorted(intervals):
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return merged


def length(merged):
    return sum(e - s for s, e in merged)


def row_metrics(row):
    """Metrics for one rank's (step, layer) record, in microseconds."""
    ivs = row["intervals"]
    by_phase = defaultdict(list)
    for iv in ivs:
        by_phase[iv["phase"]].append((iv["start_us"], iv["end_us"]))

    start = (
        min(s for s, _ in by_phase["route"])
        if by_phase["route"]
        else min(iv["start_us"] for iv in ivs)
    )
    end = (
        max(e for _, e in by_phase["post"]) if by_phase["post"] else max(iv["end_us"] for iv in ivs)
    )

    c = length(union([x for p in COMPUTE for x in by_phase[p]]))
    m = length(union([x for p in COMM for x in by_phase[p]]))
    c_or_m = length(union([x for p in COMPUTE + COMM for x in by_phase[p]]))
    makespan = end - start
    # max(0, ...) only absorbs float rounding in the set arithmetic.
    return {
        "makespan": makespan,
        "exposed_comm": max(0.0, c_or_m - c),
        "exposed_compute": max(0.0, c_or_m - m),
        "overlapped": max(0.0, c + m - c_or_m),
        "other": max(0.0, makespan - c_or_m),
        "phases": {p: sum(e - s for s, e in v) for p, v in by_phase.items()},
    }


def load(timing_dir):
    files = sorted(glob.glob(os.path.join(timing_dir, "rank*.jsonl")))
    if not files:
        sys.exit(f"no rank*.jsonl files in {timing_dir}")
    rows = []
    for f in files:
        with open(f) as fh:
            rows.extend(json.loads(line) for line in fh)
    return rows, len(files)


def stats(values):
    a = np.asarray(values, dtype=float)
    return a.mean(), np.percentile(a, 50), np.percentile(a, 99), a.max()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("timing_dir")
    ap.add_argument("--warmup", type=int, default=5, help="drop steps < K (default 5; 0 keeps all)")
    ap.add_argument(
        "--decode-max-tokens",
        type=int,
        default=64,
        help="a step is decode if every rank's n_tokens_local <= this, else prefill (default 64)",
    )
    ap.add_argument("--csv", help="write one row per (step, layer) to this file")
    ap.add_argument("--per-layer", action="store_true", help="also print mean makespan per layer")
    args = ap.parse_args()

    rows, n_ranks = load(args.timing_dir)
    groups = defaultdict(dict)  # (step, layer) -> rank -> row
    for r in rows:
        groups[(r["step"], r["layer"])][r["rank"]] = r

    # Step kind from the largest per-rank token count in that step.
    step_tokens = defaultdict(int)
    for (step, _), ranks in groups.items():
        step_tokens[step] = max(step_tokens[step], *(r["n_tokens_local"] for r in ranks.values()))
    kind_of = {
        s: ("decode" if t <= args.decode_max_tokens else "prefill") for s, t in step_tokens.items()
    }

    samples = []
    incomplete = 0
    for (step, layer), ranks in sorted(groups.items()):
        if step < args.warmup:
            continue
        if len(ranks) != n_ranks:
            incomplete += 1
            continue
        per_rank = {rank: row_metrics(r) for rank, r in ranks.items()}
        crit = max(per_rank, key=lambda k: per_rank[k]["makespan"])
        cm = per_rank[crit]
        samples.append(
            {
                "step": step,
                "layer": layer,
                "kind": kind_of[step],
                "critical_rank": crit,
                "n_tokens_local_max": step_tokens[step],
                "makespan_us": cm["makespan"],
                "exposed_comm_us": cm["exposed_comm"],
                "exposed_compute_us": cm["exposed_compute"],
                "overlapped_us": cm["overlapped"],
                "other_us": cm["other"],
                "mean_rank_exposed_comm_us": float(
                    np.mean([m["exposed_comm"] for m in per_rank.values()])
                ),
                **{f"{p}_us": d for p, d in cm["phases"].items()},
            }
        )

    paths = sorted({r["path"] for r in rows})
    steps = sorted({s for s, _ in groups})
    print(f"run: {args.timing_dir}")
    print(
        f"ranks: {n_ranks}  steps: {len(steps)} (dropped first {args.warmup})  path: {', '.join(paths)}"
    )
    print(f"step kind: decode if max n_tokens_local <= {args.decode_max_tokens}, else prefill")
    print("note: exposed comm includes waiting on peers; with TP > 1, TP comm counts as compute")
    if incomplete:
        print(f"warning: skipped {incomplete} (step, layer) groups missing some ranks")
    if not samples:
        sys.exit("no samples left after warmup")

    metrics = [
        "makespan",
        "exposed_comm",
        "exposed_compute",
        "overlapped",
        "other",
        "mean_rank_exposed_comm",
    ]
    phases = ["route", "pre", "dispatch", "compute", "combine", "post"]
    for kind in ("prefill", "decode"):
        ks = [s for s in samples if s["kind"] == kind]
        if not ks:
            print(f"\n{kind}: no samples")
            continue
        n_steps = len({s["step"] for s in ks})
        print(f"\n{kind}: {len(ks)} samples ({n_steps} steps x layers), critical rank, µs")
        print(f"  {'metric':<24}{'mean':>10}{'p50':>10}{'p99':>10}{'max':>10}{'% of W':>9}")
        mean_w = np.mean([s["makespan_us"] for s in ks])
        for m in metrics:
            vals = [s[f"{m}_us"] for s in ks]
            mean, p50, p99, mx = stats(vals)
            pct = "" if m == "makespan" else f"{100 * mean / mean_w:8.1f}%"
            print(f"  {m:<24}{mean:10.1f}{p50:10.1f}{p99:10.1f}{mx:10.1f}{pct:>9}")
        print(f"  {'phase (sum of intervals)':<24}{'mean':>10}")
        for p in phases:
            vals = [s.get(f"{p}_us", 0.0) for s in ks]
            print(f"  {p:<24}{np.mean(vals):10.1f}")

        if args.per_layer:
            by_layer = defaultdict(list)
            for s in ks:
                by_layer[s["layer"]].append(s["makespan_us"])
            print(f"  {'layer':<8}{'mean makespan':>14}{'max':>10}")
            for layer in sorted(by_layer):
                v = by_layer[layer]
                print(f"  {layer:<8}{np.mean(v):14.1f}{max(v):10.1f}")

    if args.csv:
        fields = list(dict.fromkeys(k for s in samples for k in s))
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(samples)
        print(f"\nwrote {len(samples)} rows to {args.csv}")


if __name__ == "__main__":
    main()
