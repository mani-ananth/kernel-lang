#!/usr/bin/env python3
"""Compare two Perfetto trace files and optionally merge them.

Usage:
  # Side-by-side comparison table
  python tools/compare_traces.py numpy_trace.json mlx_trace.json

  # Merge into one file for Perfetto side-by-side view
  python tools/compare_traces.py numpy_trace.json mlx_trace.json --merge comparison.json

  # Compare more than 2 traces (table shows first two, merge accepts all)
  python tools/compare_traces.py a.json b.json c.json --merge all.json
"""

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, stdev


# ── Loading ────────────────────────────────────────────────────────────────────

def load(path: str) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    return data.get("traceEvents", data if isinstance(data, list) else [])


def x_events(events: list[dict]) -> dict[str, list[float]]:
    """Group durations of complete (X) events by name."""
    result: dict[str, list[float]] = {}
    for ev in events:
        if ev.get("ph") == "X":
            result.setdefault(ev["name"], []).append(ev["dur"])
    return result


# ── Comparison table ───────────────────────────────────────────────────────────

def compare(path1: str, path2: str):
    name1 = Path(path1).stem
    name2 = Path(path2).stem

    d1 = x_events(load(path1))
    d2 = x_events(load(path2))

    all_names = sorted(
        set(d1) | set(d2),
        key=lambda n: -(max(mean(d1.get(n, [0])), mean(d2.get(n, [0]))))
    )

    W_NAME = 26
    W_COL  = 18
    print(f"\n{'Event':<{W_NAME}} | {name1:^{W_COL}} | {name2:^{W_COL}} | Speedup")
    print("─" * (W_NAME + 2 * W_COL + 20))

    for name in all_names:
        v1 = d1.get(name)
        v2 = d2.get(name)

        def fmt(vals):
            if not vals:
                return f"{'—':^{W_COL}}"
            mu, sd = mean(vals), stdev(vals) if len(vals) > 1 else 0
            return f"{mu:>7.1f}μs ±{sd:>5.1f}"

        s1 = fmt(v1)
        s2 = fmt(v2)

        if v1 and v2:
            mu1, mu2 = mean(v1), mean(v2)
            ratio = mu1 / mu2
            if ratio > 1:
                sp = f"{ratio:.1f}x  {name2} faster"
            else:
                sp = f"{1/ratio:.1f}x  {name1} faster"
        else:
            sp = "—"

        print(f"{name:<{W_NAME}} | {s1} | {s2} | {sp}")

    print()

    # Totals for the "kernel_call" span if present, else sum of X events
    def total(d):
        if "kernel_call" in d:
            return mean(d["kernel_call"])
        return sum(mean(v) for v in d.values())

    t1, t2 = total(d1), total(d2)
    winner = name1 if t1 < t2 else name2
    ratio  = max(t1, t2) / min(t1, t2)
    print(f"  Overall: {winner} is {ratio:.1f}x faster  ({t1:.1f}μs vs {t2:.1f}μs)")


# ── Merge ──────────────────────────────────────────────────────────────────────

def merge(paths: list[str], output: str):
    """Merge multiple traces into one file. Each file gets its own PID."""
    all_events: list[dict] = []

    for i, path in enumerate(paths, start=1):
        name = Path(path).stem
        events = load(path)
        # Override process name metadata
        all_events.append({
            "ph": "M", "name": "process_name",
            "pid": i, "tid": 0, "args": {"name": f"[{i}] {name}"}
        })
        for ev in events:
            e = dict(ev)
            e["pid"] = i
            # Skip original process_name metadata (we just added our own)
            if e.get("ph") == "M" and e.get("name") == "process_name":
                continue
            all_events.append(e)

    with open(output, "w") as f:
        json.dump({"traceEvents": all_events}, f)

    print(f"Merged {len(paths)} traces ({len(all_events)} events) → {output}")
    print("  Open in ui.perfetto.dev to view side-by-side")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare and merge Perfetto Chrome Trace JSON files"
    )
    parser.add_argument("traces", nargs="+", metavar="trace.json",
                        help="Trace files to compare/merge")
    parser.add_argument("--merge", metavar="OUTPUT",
                        help="Merge all traces into a single file for Perfetto")
    args = parser.parse_args()

    if len(args.traces) < 1:
        parser.error("Provide at least one trace file")

    if len(args.traces) >= 2:
        compare(args.traces[0], args.traces[1])

    if args.merge:
        merge(args.traces, args.merge)
    elif len(args.traces) == 1:
        # Single file: just print event summary
        d = x_events(load(args.traces[0]))
        print(f"\n{Path(args.traces[0]).stem}: {sum(len(v) for v in d.values())} X-events\n")
        for name, vals in sorted(d.items(), key=lambda kv: -mean(kv[1])):
            mu = mean(vals)
            print(f"  {name:<28}  {mu:>8.1f}μs  (n={len(vals)})")


if __name__ == "__main__":
    main()
