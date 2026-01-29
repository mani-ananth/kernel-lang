#!/usr/bin/env python3
"""Benchmark: matrix multiplication via mini_pallas vs raw NumPy."""

import statistics
import time

import numpy as np

import mini_pallas
from mini_pallas.trace import trace_kernel
from mini_pallas.runtime import compile_numpy, _cache


@mini_pallas.kernel
def matmul_kernel(a_ref, b_ref, o_ref):
  a, b = a_ref[...], b_ref[...]
  o_ref[...] = a @ b


def bench(fn, rounds=10, warmup=2):
  """Run fn, return list of elapsed times (seconds) after warmup."""
  for _ in range(warmup):
    fn()
  times = []
  for _ in range(rounds):
    t0 = time.perf_counter()
    fn()
    times.append(time.perf_counter() - t0)
  return times


def fmt(times):
  """Format timing stats: median (min–max) in ms or µs."""
  med = statistics.median(times)
  lo, hi = min(times), max(times)
  if med >= 1e-3:
    return f"{med*1e3:8.3f} ms  (min {lo*1e3:.3f}, max {hi*1e3:.3f})"
  return f"{med*1e6:8.1f} µs  (min {lo*1e6:.1f}, max {hi*1e6:.1f})"


SIZES = [64, 256, 1024, 2048]
ROUNDS = 20


def main():
  print("Mini Pallas — Matrix Multiply Benchmark")
  print("=" * 60)

  # --- fixed overhead: trace + compile (size-independent) ---
  print("\n[Trace + Compile overhead]")

  trace_times = bench(lambda: trace_kernel(matmul_kernel._fn), rounds=ROUNDS)
  print(f"  Trace:    {fmt(trace_times)}")

  ir = trace_kernel(matmul_kernel._fn)
  def compile_fresh():
    _cache.clear()
    compile_numpy(ir)
  compile_times = bench(compile_fresh, rounds=ROUNDS)
  print(f"  Compile:  {fmt(compile_times)}")

  # --- per-size execution ---
  print(f"\n[Execution]  ({ROUNDS} rounds, median)")
  print(f"  {'Size':>6s}  {'NumPy':>22s}  {'Kernel':>22s}  {'Overhead':>8s}")
  print(f"  {'-'*6}  {'-'*22}  {'-'*22}  {'-'*8}")

  for n in SIZES:
    a = np.random.randn(n, n)
    b = np.random.randn(n, n)
    out_np = np.empty((n, n))
    out_kn = np.empty((n, n))

    # raw numpy
    np_times = bench(lambda: np.matmul(a, b, out=out_np), rounds=ROUNDS)

    # mini_pallas kernel (warm — already traced+compiled)
    matmul_kernel(a, b, out_kn)  # ensure warm
    kn_times = bench(lambda: matmul_kernel(a, b, out_kn), rounds=ROUNDS)

    np_med = statistics.median(np_times)
    kn_med = statistics.median(kn_times)
    overhead = (kn_med / np_med - 1) * 100 if np_med > 0 else float('inf')

    np_s = fmt(np_times)
    kn_s = fmt(kn_times)
    print(f"  {n:>5d}   {np_s}  {kn_s}  {overhead:+.1f}%")

    # sanity check
    assert np.allclose(out_np, out_kn), f"Mismatch at size {n}!"

  # --- cold vs warm call ---
  print(f"\n[Cold vs Warm call]  (size=1024, {ROUNDS} rounds, median)")
  n = 1024
  a = np.random.randn(n, n)
  b = np.random.randn(n, n)
  out = np.empty((n, n))

  def cold_call():
    kf = mini_pallas.KernelFunction(matmul_kernel._fn)
    _cache.clear()
    kf(a, b, out)

  cold_times = bench(cold_call, rounds=ROUNDS)
  warm_times = bench(lambda: matmul_kernel(a, b, out), rounds=ROUNDS)

  print(f"  Cold:  {fmt(cold_times)}")
  print(f"  Warm:  {fmt(warm_times)}")


if __name__ == "__main__":
  main()
