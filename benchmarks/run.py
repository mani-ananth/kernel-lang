#!/usr/bin/env python3
"""Generic kernel benchmark runner.

Usage:
  python benchmarks/run.py --kernel=examples/02_matrix_multiply.py::matmul_kernel
  python benchmarks/run.py --kernel=examples/01_vector_add.py::add_vectors_kernel --size=10000 --ndim=1
  python benchmarks/run.py --kernel=examples/02_matrix_multiply.py::matmul_kernel --size=512 --rounds=50
"""

import argparse
import importlib.util
import inspect
import statistics
import sys
import time

import numpy as np

import mini_pallas
from mini_pallas.trace import trace_kernel
from mini_pallas.runtime import compile_numpy, _cache


def load_kernel(spec):
  """Load a kernel from a 'path/to/file.py::function_name' specifier."""
  if "::" not in spec:
    print(f"Error: expected 'file.py::function_name', got '{spec}'", file=sys.stderr)
    sys.exit(1)
  path, name = spec.rsplit("::", 1)
  mod_spec = importlib.util.spec_from_file_location("_bench_mod", path)
  mod = importlib.util.module_from_spec(mod_spec)
  mod_spec.loader.exec_module(mod)
  obj = getattr(mod, name, None)
  if obj is None:
    print(f"Error: '{name}' not found in {path}", file=sys.stderr)
    sys.exit(1)
  if not isinstance(obj, mini_pallas.KernelFunction):
    print(f"Error: '{name}' is not a mini_pallas.kernel", file=sys.stderr)
    sys.exit(1)
  return obj


def make_arrays(kernel, size, ndim, dtype):
  """Generate random input arrays and an output array from the kernel signature."""
  sig = inspect.signature(kernel._fn)
  n_params = len(sig.parameters)
  n_inputs = n_params - 1  # last param is output ref
  shape = (size,) * ndim
  inputs = [np.random.randn(*shape).astype(dtype) for _ in range(n_inputs)]
  output = np.empty(shape, dtype=dtype)
  return inputs, output


def bench(fn, rounds=20, warmup=2):
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
  """Format timing stats: median (min-max) in ms or us."""
  med = statistics.median(times)
  lo, hi = min(times), max(times)
  if med >= 1e-3:
    return f"{med*1e3:8.3f} ms  (min {lo*1e3:.3f}, max {hi*1e3:.3f})"
  return f"{med*1e6:8.1f} us  (min {lo*1e6:.1f}, max {hi*1e6:.1f})"


def main():
  parser = argparse.ArgumentParser(description="Benchmark a mini_pallas kernel")
  parser.add_argument("--kernel", required=True,
    help="file.py::function_name")
  parser.add_argument("--size", type=int, default=1024,
    help="array dimension size (default: 1024)")
  parser.add_argument("--ndim", type=int, default=2,
    help="number of array dimensions (default: 2)")
  parser.add_argument("--dtype", default="float64",
    help="numpy dtype (default: float64)")
  parser.add_argument("--rounds", type=int, default=20,
    help="timed rounds (default: 20)")
  parser.add_argument("--warmup", type=int, default=2,
    help="warmup rounds (default: 2)")
  args = parser.parse_args()

  dtype = np.dtype(args.dtype)
  kernel = load_kernel(args.kernel)
  inputs, output = make_arrays(kernel, args.size, args.ndim, dtype)
  shape_str = "x".join(str(args.size) for _ in range(args.ndim))

  sig = inspect.signature(kernel._fn)
  param_names = list(sig.parameters.keys())

  print(f"Kernel:  {kernel._fn.__name__}")
  print(f"Params:  {', '.join(param_names)}")
  print(f"Shape:   {shape_str}  dtype={dtype}")
  print(f"Rounds:  {args.rounds}  warmup={args.warmup}")
  print("=" * 55)

  # trace
  trace_times = bench(
    lambda: trace_kernel(kernel._fn),
    rounds=args.rounds, warmup=args.warmup,
  )
  print(f"  Trace:    {fmt(trace_times)}")

  #  compile 
  ir = trace_kernel(kernel._fn)
  def compile_fresh():
    _cache.clear()
    compile_numpy(ir)
  compile_times = bench(compile_fresh, rounds=args.rounds, warmup=args.warmup)
  print(f"  Compile:  {fmt(compile_times)}")

  #  execute (warm) 
  kernel(*inputs, output)  # prime cache
  exec_times = bench(
    lambda: kernel(*inputs, output),
    rounds=args.rounds, warmup=args.warmup,
  )
  print(f"  Execute:  {fmt(exec_times)}")

  #  cold (trace + compile + execute) 
  def cold_call():
    kf = mini_pallas.KernelFunction(kernel._fn)
    _cache.clear()
    kf(*inputs, output)
  cold_times = bench(cold_call, rounds=args.rounds, warmup=args.warmup)
  print(f"  Cold:     {fmt(cold_times)}")

  #  summary 
  trace_med = statistics.median(trace_times)
  compile_med = statistics.median(compile_times)
  exec_med = statistics.median(exec_times)
  cold_med = statistics.median(cold_times)
  print()
  print(f"  Overhead (trace+compile): {(trace_med + compile_med)*1e6:.1f} us")
  print(f"  Cold / Warm ratio:        {cold_med / exec_med:.3f}x")


if __name__ == "__main__":
  main()
