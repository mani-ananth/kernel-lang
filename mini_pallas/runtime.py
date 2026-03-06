"""Execution: compile lowered source and run it."""

from typing import Callable

import numpy as np

from .core import KernelIR
from .lowering import lower_fused_numpy, lower_to_numpy

_cache: dict[int, tuple[KernelIR, Callable]] = {}
_fused_cache: dict[int, tuple[KernelIR, Callable]] = {}


def compile_numpy(ir: KernelIR) -> Callable:
  """Compile a KernelIR to a callable Python/NumPy function (unfused)."""
  key = id(ir)
  if key in _cache and _cache[key][0] is ir:
    return _cache[key][1]

  source = lower_to_numpy(ir)
  namespace: dict = {"np": np}  # make numpy available in generated code
  exec(source, namespace)
  fn = namespace[ir.name]
  _cache[key] = (ir, fn)
  return fn


def compile_fused_numpy(ir: KernelIR) -> Callable:
  """Compile a KernelIR to a callable Python/NumPy function with fusion."""
  key = id(ir)
  if key in _fused_cache and _fused_cache[key][0] is ir:
    return _fused_cache[key][1]

  source = lower_fused_numpy(ir)
  namespace: dict = {"np": np}
  exec(source, namespace)
  fn = namespace[ir.name]
  _fused_cache[key] = (ir, fn)
  return fn
