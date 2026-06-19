"""Tests for mini_pallas.runtime — compilation and caching."""

from mini_pallas.runtime import compile_numpy, _numpy_cache
from mini_pallas.trace import trace_kernel


def test_compile_returns_callable():
  def k(x, o):
    o[...] = x[...]

  ir = trace_kernel(k)
  _numpy_cache.clear()
  assert callable(compile_numpy(ir))


def test_compile_caching():
  """Same IR object returns the same function from cache."""
  def k(x, o):
    o[...] = x[...]

  ir = trace_kernel(k)
  _numpy_cache.clear()
  fn1 = compile_numpy(ir)
  fn2 = compile_numpy(ir)
  assert fn1 is fn2


def test_compile_cache_repopulated_after_clear():
  def k(x, o):
    o[...] = x[...]

  ir = trace_kernel(k)
  _numpy_cache.clear()
  compile_numpy(ir)
  _numpy_cache.clear()
  compile_numpy(ir)
  assert id(ir) in _numpy_cache
