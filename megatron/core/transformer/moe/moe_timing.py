"""Per-phase CUDA-event timing for MoE layers.

Enabled by setting the ``MOE_PHASE_TIMING_DIR`` environment variable to an output directory.
Each rank writes one JSON line per (step, MoE layer) to ``<dir>/rank<NNNNN>.jsonl``.

Events are recorded on whichever stream runs the timed work, so overlapping phases on different
streams are measured without any host synchronization in the hot path. Events are resolved
lazily: at the start of each step, completed records from earlier steps are written out.

See ``deliverance/docs/moe-timing.md`` for the schema and the metrics derived from it.
"""

import atexit
import json
import os
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Iterator, List, Optional

import torch

from megatron.core.utils import nvtx_range_pop, nvtx_range_push

MOE_PHASE_TIMING_ENV = "MOE_PHASE_TIMING_DIR"


@dataclass
class _Interval:
    phase: str
    idx: Optional[int]
    tokens: Optional[int]
    start: torch.cuda.Event
    end: Optional[torch.cuda.Event] = None


@dataclass
class LayerTimingRecord:
    """Timing state of one MoE layer forward pass on one rank."""

    timer: "MoEPhaseTimer"
    step: int
    layer: int
    path: str
    n_tokens_local: int
    ref: torch.cuda.Event
    n_tokens_recv: Optional[int] = None
    schedule_cpu_us: Optional[float] = None
    intervals: List[_Interval] = field(default_factory=list)

    @contextmanager
    def phase(
        self, name: str, idx: Optional[int] = None, tokens: Optional[int] = None
    ) -> Iterator[_Interval]:
        """Time the enclosed work on the current CUDA stream.

        The yielded interval's ``tokens`` may be set inside the block when the count is only
        known after the work is issued. On exception the interval is discarded.
        """
        tag = f"moe.{name}" if idx is None else f"moe.{name}[{idx}]"
        nvtx_range_push(tag)
        try:
            interval = _Interval(name, idx, tokens, self.timer._record_event())
            yield interval
            interval.end = self.timer._record_event()
            self.intervals.append(interval)
        finally:
            nvtx_range_pop(tag)


def moe_phase(
    record: Optional[LayerTimingRecord],
    name: str,
    idx: Optional[int] = None,
    tokens: Optional[int] = None,
):
    """``record.phase(...)`` when timing is active, otherwise a no-op context.

    The no-op context yields a throwaway object so callers can set ``.tokens`` unconditionally.
    """
    if record is None:
        return nullcontext(SimpleNamespace(tokens=None))
    return record.phase(name, idx, tokens)


class MoEPhaseTimer:
    """Collects per-phase CUDA-event timings of MoE layers and writes them as JSON lines."""

    def __init__(self, out_dir: str, rank: int) -> None:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"rank{rank:05d}.jsonl")
        try:
            self._file = open(path, "x")
        except FileExistsError as e:
            raise FileExistsError(
                f"{path} already exists; point {MOE_PHASE_TIMING_ENV} at a fresh directory"
            ) from e
        self.rank = rank
        self._pool: List[torch.cuda.Event] = []
        self._pending: List[LayerTimingRecord] = []
        self._first_layer: Optional[int] = None
        self._step = -1

    def _record_event(self) -> torch.cuda.Event:
        event = self._pool.pop() if self._pool else torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    def begin_layer(
        self, layer: int, path: str, n_tokens_local: int
    ) -> Optional[LayerTimingRecord]:
        """Start timing one MoE layer forward pass. Returns None while a CUDA graph is captured.

        The step counter advances whenever the first layer seen by this timer starts again.
        """
        if torch.cuda.is_current_stream_capturing():
            return None
        if self._first_layer is None:
            self._first_layer = layer
        if layer == self._first_layer:
            self._step += 1
            self._drain(block=False)
        return LayerTimingRecord(
            timer=self,
            step=self._step,
            layer=layer,
            path=path,
            n_tokens_local=n_tokens_local,
            ref=self._record_event(),
        )

    def end_layer(self, record: Optional[LayerTimingRecord]) -> None:
        """Queue a finished layer record for resolution."""
        if record is not None:
            self._pending.append(record)

    def _drain(self, block: bool) -> None:
        """Write out pending records in order, stopping at the first one still running on the GPU."""
        done = 0
        for record in self._pending:
            events = [record.ref]
            for interval in record.intervals:
                events += [interval.start, interval.end]
            if block:
                for event in events:
                    event.synchronize()
            elif not all(event.query() for event in events):
                break
            self._file.write(json.dumps(self._to_row(record)) + "\n")
            self._pool.extend(events)
            done += 1
        if done:
            del self._pending[:done]
            self._file.flush()

    def _to_row(self, record: LayerTimingRecord) -> dict:
        def offset_us(event: torch.cuda.Event) -> float:
            return round(record.ref.elapsed_time(event) * 1000.0, 3)

        return {
            "rank": self.rank,
            "step": record.step,
            "layer": record.layer,
            "path": record.path,
            "n_tokens_local": record.n_tokens_local,
            "n_tokens_recv": record.n_tokens_recv,
            "schedule_cpu_us": record.schedule_cpu_us,
            "intervals": [
                {
                    "phase": interval.phase,
                    "idx": interval.idx,
                    "tokens": interval.tokens,
                    "start_us": offset_us(interval.start),
                    "end_us": offset_us(interval.end),
                }
                for interval in record.intervals
            ],
        }

    def close(self) -> None:
        """Resolve every pending record (blocking) and close the output file."""
        if self._file.closed:
            return
        self._drain(block=True)
        self._file.close()


_timer: Optional[MoEPhaseTimer] = None
_timer_checked = False


def get_moe_phase_timer() -> Optional[MoEPhaseTimer]:
    """Return the process-wide timer, or None if ``MOE_PHASE_TIMING_DIR`` is unset."""
    global _timer, _timer_checked
    if not _timer_checked:
        _timer_checked = True
        out_dir = os.environ.get(MOE_PHASE_TIMING_ENV)
        if out_dir:
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            _timer = MoEPhaseTimer(out_dir, rank)
            atexit.register(_timer.close)
    return _timer
