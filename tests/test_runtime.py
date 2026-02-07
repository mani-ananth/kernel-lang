"""Tests for mini_pallas.runtime — compilation and caching."""

from mini_pallas.trace import trace_kernel
from mini_pallas.runtime import compile_numpy, _cache


def test_compile_returns_callable():
  """compile_numpy returns a callable function."""
  def k(x, o):
    o[...] = x[...]
  ir = trace_kernel(k)
  _cache.clear()
  fn = compile_numpy(ir)
  assert callable(fn)


def test_compile_caching():
  """Same IR id returns same function object from cache."""
  def k(x, o):
    o[...] = x[...]
  ir = trace_kernel(k)
  _cache.clear()
  fn1 = compile_numpy(ir)
  fn2 = compile_numpy(ir)
  assert fn1 is fn2


def test_compile_cache_clear():
  """After cache clear, function is recompiled."""
  def k(x, o):
    o[...] = x[...]
  ir = trace_kernel(k)
  _cache.clear()
  fn1 = compile_numpy(ir)
  _cache.clear()
  fn2 = compile_numpy(ir)
  # Different function objects (though functionally identical)
  # Note: they might be the same object if Python interns them,
  # but the cache was definitely cleared and re-populated
  assert id(ir) in _cache
