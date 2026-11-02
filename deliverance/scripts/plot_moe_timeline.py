"""Plot MoE phase-timing intervals as a per-stream timeline (one PNG per step and layer).

Lanes are CUDA streams:

  main      route, pre, post (and, on the stock path, dispatch / compute / combine)
  dispatch  dispatch[k], k >= 1 (one grouped P2P per matching)
  compute   dispatch[0] (local-chunk copy) and compute[b]
  combine   combine[k]

Times are microseconds from each rank's own layer reference event. Ranks share no clock, so
subplots of different ranks line up at their layer start, not in absolute time.

Usage (from <MEGATRON_ROOT>, venv active):
  python deliverance/scripts/plot_moe_timeline.py local/moe-timing/<run> --step 9 --layer 17
  python deliverance/scripts/plot_moe_timeline.py local/moe-timing/<run> --step 0 9 --layer 1 17 \
      --ranks 0 1 2 3 --zoom
"""

import argparse
import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

LANES = ("main", "dispatch", "compute", "combine")
# Categorical slots 1-3 of the validated reference palette; main-stream phases in neutral grey.
COLORS = {"dispatch": "#2a78d6", "compute": "#eb6834", "combine": "#1baf7a", "main": "#a3a29c"}
SURFACE = "#fcfcfb"
TEXT, TEXT_2 = "#0b0b0b", "#52514e"


def lane_of(interval, decomposed):
    phase, idx = interval["phase"], interval["idx"]
    if not decomposed or phase not in ("dispatch", "compute", "combine"):
        return "main"
    if phase == "dispatch" and idx == 0:
        return "compute"  # the local-chunk copy runs on the compute stream
    return phase


def draw_rank(ax, row, xlim, zoom):
    decomposed = any(iv["idx"] is not None for iv in row["intervals"])
    lanes = LANES if decomposed else ("main",)
    if xlim:
        x0, x1 = xlim
    elif zoom:
        # This rank's dispatch -> combine section, with a small margin.
        moe = [iv for iv in row["intervals"] if iv["phase"] in ("dispatch", "compute", "combine")]
        x0, x1 = min(iv["start_us"] for iv in moe), max(iv["end_us"] for iv in moe)
        pad = 0.02 * (x1 - x0)
        x0, x1 = x0 - pad, x1 + pad
    else:
        x0, x1 = 0.0, max(iv["end_us"] for iv in row["intervals"])
    for i, lane in enumerate(lanes):
        y = len(lanes) - 1 - i
        for iv in row["intervals"]:
            if lane_of(iv, decomposed) != lane:
                continue
            start, width = iv["start_us"], iv["end_us"] - iv["start_us"]
            color = COLORS.get(iv["phase"], COLORS["main"]) if decomposed or iv["phase"] in COLORS else COLORS["main"]
            ax.barh(y, width, left=start, height=0.6, color=color, edgecolor=SURFACE, linewidth=1)
            # Label each bar with its matching / batch index (or phase name) when it fits.
            label = iv["phase"] if iv["idx"] is None else str(iv["idx"])
            if width > 0.03 * (x1 - x0) and x0 <= start + width / 2 <= x1:
                ax.text(start + width / 2, y, label, ha="center", va="center", fontsize=7, color="white")
    ax.set_yticks(range(len(lanes)), labels=list(reversed(lanes)), color=TEXT)
    ax.set_xlim(x0, x1)
    # Extra tick at the final time: the end of the layer, or with --zoom the end of the last
    # dispatch/compute/combine interval. Regular ticks close enough to collide are dropped.
    shown = [
        iv for iv in row["intervals"]
        if (not zoom or xlim or iv["phase"] in ("dispatch", "compute", "combine"))
        and iv["start_us"] <= x1
    ]
    end = min(max(iv["end_us"] for iv in shown), x1)
    ticks = [t for t in ax.get_xticks() if x0 <= t <= x1 and abs(t - end) > 0.06 * (x1 - x0)]
    ax.set_xticks(ticks + [end], labels=[f"{t:g}" for t in ticks] + [f"{end:.0f}"])
    ax.get_xticklabels()[-1].set_fontweight("bold")
    ax.grid(axis="x", color="#e4e3df", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=TEXT_2, labelsize=8)
    sched = row.get("schedule_cpu_us")
    ax.set_title(
        f"rank {row['rank']} · {row['path']} · tokens local {row['n_tokens_local']}, "
        f"received {row['n_tokens_recv']}" + (f" · schedule CPU {sched:.0f} µs" if sched else ""),
        loc="left", fontsize=9, color=TEXT_2,
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("timing_dir")
    parser.add_argument("--step", type=int, nargs="+", required=True)
    parser.add_argument("--layer", type=int, nargs="+", required=True)
    parser.add_argument("--ranks", nargs="+", default=["0"], help="rank ids, or 'all'")
    parser.add_argument(
        "--zoom", action="store_true",
        help="crop each rank to its own first-dispatch .. last-combine window",
    )
    parser.add_argument(
        "--xlim", type=float, nargs=2,
        help="explicit window in µs, applied to every rank (their clocks differ)",
    )
    parser.add_argument("-o", "--out-dir", help="output directory (default: the timing dir)")
    parser.add_argument(
        "--prefix", default="timeline",
        help="output file name prefix: <prefix>_step<S>_layer<L>.png (default: timeline)",
    )
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.timing_dir, "rank*.jsonl")))
    if not files:
        sys.exit(f"no rank*.jsonl in {args.timing_dir}")
    rows = [json.loads(line) for f in files for line in open(f)]
    ranks = None if args.ranks == ["all"] else {int(r) for r in args.ranks}
    out_dir = args.out_dir or args.timing_dir
    os.makedirs(out_dir, exist_ok=True)

    for step in args.step:
        for layer in args.layer:
            picked = sorted(
                (r for r in rows if r["step"] == step and r["layer"] == layer
                 and (ranks is None or r["rank"] in ranks)),
                key=lambda r: r["rank"],
            )
            if not picked:
                print(f"no rows for step {step} layer {layer}", file=sys.stderr)
                continue
            fig, axes = plt.subplots(
                len(picked), 1, figsize=(14, 1.9 * len(picked) + 0.8), squeeze=False,
                facecolor=SURFACE,
            )
            for ax, row in zip(axes[:, 0], picked):
                ax.set_facecolor(SURFACE)
                draw_rank(ax, row, args.xlim, args.zoom)
            axes[-1, 0].set_xlabel("µs from the rank's layer start (ranks share no clock)", color=TEXT_2)
            fig.legend(
                handles=[Patch(color=COLORS[p], label=p) for p in ("dispatch", "compute", "combine")]
                + [Patch(color=COLORS["main"], label="route / pre / post")],
                loc="upper right", ncol=4, frameon=False, fontsize=9,
            )
            fig.suptitle(f"step {step} · layer {layer}", x=0.01, ha="left", fontsize=12, color=TEXT)
            fig.tight_layout(rect=(0, 0, 1, 0.96))
            path = os.path.join(out_dir, f"{args.prefix}_step{step}_layer{layer}.png")
            fig.savefig(path, dpi=130)
            plt.close(fig)
            print(path)


if __name__ == "__main__":
    main()
