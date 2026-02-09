"""Execution: compile lowered source and run it."""

from typing import Callable

import numpy as np

from .core import KernelIR
from .lowering import lower_to_numpy

_cache: dict[int, Callable] = {}


def compile_numpy(ir: KernelIR) -> Callable:
  """Compile a KernelIR to a callable Python/NumPy function."""
  key = id(ir)
  if key in _cache:
    return _cache[key]

  source = lower_to_numpy(ir)
  namespace: dict = {"np": np}  # make numpy available in generated code
  exec(source, namespace)
  fn = namespace[ir.name]
  _cache[key] = fn
  return fn
