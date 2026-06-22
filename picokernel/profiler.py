"""Lightweight Perfetto-compatible profiler using Chrome Trace Event format.

Output JSON files can be loaded directly at ui.perfetto.dev.
"""

import json
import resource
import time
from contextlib import contextmanager


class Profiler:
    """Records events in Chrome Trace Event (JSON) format.

    ts and dur values are in microseconds, which is what Perfetto expects.
    """

    def __init__(self, name: str = "trace", pid: int = 1, tid: int = 1):
        self.name = name
        self._pid = pid
        self._tid = tid
        self._events: list[dict] = []
        self._t0 = time.perf_counter()
        self._meta("process_name", pid, 0, {"name": name})
        self._meta("thread_name", pid, tid, {"name": "main"})

    # ------------------------------------------------------------------
    # Internal helpers

    def _ts(self) -> float:
        """Microseconds elapsed since this profiler was created."""
        return (time.perf_counter() - self._t0) * 1e6

    def _meta(self, name: str, pid: int, tid: int, args: dict):
        self._events.append({"ph": "M", "name": name, "pid": pid, "tid": tid, "args": args})

    # ------------------------------------------------------------------
    # Public event API

    def complete(self, name: str, ts: float, dur: float, cat: str = "op",
                 pid: int | None = None, tid: int | None = None, **args):
        """Emit a complete (X) event — preferred for generated code."""
        ev = {
            "ph": "X", "name": name, "cat": cat,
            "ts": ts,
            "dur": max(dur, 0.001),  # Perfetto requires dur > 0
            "pid": pid if pid is not None else self._pid,
            "tid": tid if tid is not None else self._tid,
        }
        if args:
            ev["args"] = args
        self._events.append(ev)

    def counter(self, name: str, values: dict, pid: int | None = None):
        """Emit a counter (C) event — visible as a graph track in Perfetto."""
        self._events.append({
            "ph": "C", "name": name, "ts": self._ts(),
            "pid": pid if pid is not None else self._pid,
            "tid": self._tid,
            "args": values,
        })

    @contextmanager
    def span(self, name: str, cat: str = "op",
             pid: int | None = None, tid: int | None = None, **args):
        """Context manager emitting B/E events — use for outer scopes."""
        pid = pid if pid is not None else self._pid
        tid = tid if tid is not None else self._tid
        begin: dict = {"ph": "B", "name": name, "cat": cat, "ts": self._ts(), "pid": pid, "tid": tid}
        if args:
            begin["args"] = args
        self._events.append(begin)
        try:
            yield self
        finally:
            self._events.append({"ph": "E", "name": name, "cat": cat, "ts": self._ts(), "pid": pid, "tid": tid})

    # ------------------------------------------------------------------
    # System counter snapshots

    def record_counters(self):
        """Snapshot RSS and page-fault counters from the OS."""
        ru = resource.getrusage(resource.RUSAGE_SELF)
        # macOS: ru_maxrss is in bytes; Linux: KB
        import platform
        rss_bytes = ru.ru_maxrss if platform.system() == "Darwin" else ru.ru_maxrss * 1024
        self.counter("rss_mb", {"rss_mb": round(rss_bytes / 1024 / 1024, 2)})
        self.counter("page_faults", {"minor": ru.ru_minflt, "major": ru.ru_majflt})

    def record_mlx_memory(self):
        """Snapshot GPU memory counters from MLX Metal."""
        try:
            import mlx.core as mx
            self.counter("gpu_memory_mb", {
                "active_mb": round(mx.get_active_memory() / 1024 / 1024, 2),
                "peak_mb": round(mx.get_peak_memory() / 1024 / 1024, 2),
            })
        except (ImportError, AttributeError):
            pass

    # ------------------------------------------------------------------
    # Output

    def save(self, path: str):
        """Write a Chrome Trace Event JSON file loadable in ui.perfetto.dev."""
        with open(path, "w") as f:
            json.dump({"traceEvents": self._events}, f)
        print(f"  saved {len(self._events)} events → {path}")

    # ------------------------------------------------------------------
    # Merging

    @classmethod
    def merge(cls, *profilers: "Profiler", name: str = "comparison") -> "Profiler":
        """Merge profilers into one file, assigning each its own PID (timeline row).

        Timestamps are kept relative to each profiler's own start, so both
        timelines begin at t=0 — making duration comparison easy in Perfetto.
        """
        merged = cls.__new__(cls)
        merged.name = name
        merged._pid = 1
        merged._tid = 1
        merged._t0 = time.perf_counter()
        merged._events = []

        for i, p in enumerate(profilers, start=1):
            for ev in p._events:
                e = dict(ev)
                e["pid"] = i
                merged._events.append(e)

        return merged
