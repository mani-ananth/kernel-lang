"""Tests for guard-based retracing in KernelFunction.

Each unique (shapes, dtypes) signature maintains a separate cache entry.
Guards are checked on every call; a miss triggers retrace + recompile.
"""

import numpy as np
import pytest

import mini_pallas


def test_different_shapes_cached_separately():
  """Calling with two different shapes creates two independent cache entries."""
  @mini_pallas.kernel
  def k(x, o):
    o[...] = x[...] * 2.0

  x1, out1 = np.array([1.0, 2.0, 3.0]), np.zeros(3)
  x2, out2 = np.array([1.0, 2.0, 3.0, 4.0, 5.0]), np.zeros(5)
  k(x1, out1)
  k(x2, out2)

  assert len(k._guard_cache) == 2


def test_same_shape_uses_cache():
  """Repeated calls with the same shape use the cached entry (one entry)."""
  @mini_pallas.kernel
  def k(x, o):
    o[...] = x[...] * 2.0

  x, out = np.array([1.0, 2.0, 3.0]), np.zeros(3)
  k(x, out)
  k(x, out)

  assert len(k._guard_cache) == 1


def test_different_dtypes_cached_separately():
  """Same shape but different dtypes produces two independent cache entries."""
  @mini_pallas.kernel
  def k(x, o):
    o[...] = x[...]

  x_f32 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
  out_f32 = np.zeros(3, dtype=np.float32)
  k(x_f32, out_f32)

  x_f64 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
  out_f64 = np.zeros(3, dtype=np.float64)
  k(x_f64, out_f64)

  assert len(k._guard_cache) == 2


def test_alternating_shapes_no_retrace():
  """Alternating between two shapes reuses existing entries; cache stays at 2."""
  @mini_pallas.kernel
  def k(x, o):
    o[...] = x[...] * 2.0

  x1, out1 = np.array([1.0, 2.0, 3.0]), np.zeros(3)
  x2, out2 = np.array([1.0, 2.0, 3.0, 4.0, 5.0]), np.zeros(5)

  k(x1, out1)  # miss  → trace (3,)
  k(x2, out2)  # miss  → trace (5,)
  k(x1, out1)  # hit   → reuse (3,)
  k(x2, out2)  # hit   → reuse (5,)

  assert len(k._guard_cache) == 2


def test_guard_hit_correctness():
  """Guard hits produce the same correct results as the initial trace."""
  @mini_pallas.kernel
  def k(x, o):
    o[...] = x[...] * 3.0

  x2, out2 = np.array([1.0, 2.0]), np.zeros(2)
  x3, out3 = np.array([1.0, 2.0, 3.0]), np.zeros(3)

  k(x2, out2)
  np.testing.assert_array_equal(out2, [3.0, 6.0])

  k(x3, out3)
  np.testing.assert_array_equal(out3, [3.0, 6.0, 9.0])

  # Re-run with shape (2,) — guard hit, should still be correct
  out2b = np.zeros(2)
  k(x2, out2b)
  np.testing.assert_array_equal(out2b, [3.0, 6.0])


def test_cache_isolated_per_kernel_instance():
  """Two separate @kernel instances have independent guard caches."""
  @mini_pallas.kernel
  def k1(x, o):
    o[...] = x[...]

  @mini_pallas.kernel
  def k2(x, o):
    o[...] = x[...]

  x, out = np.array([1.0, 2.0, 3.0]), np.zeros(3)
  k1(x, out)

  assert len(k1._guard_cache) == 1
  assert len(k2._guard_cache) == 0


def test_guard_key_covers_all_params():
  """Guard key distinguishes different shapes across multiple parameters."""
  @mini_pallas.kernel
  def k(x, y, o):
    o[...] = x[...] + y[...]

  x1 = np.array([1.0, 2.0, 3.0])
  y1 = np.array([4.0, 5.0, 6.0])
  out1 = np.zeros(3)
  k(x1, y1, out1)

  # Same shapes again — should hit cache
  k(x1, y1, out1)
  assert len(k._guard_cache) == 1


def test_many_shapes_all_cached():
  """N different shapes result in N cache entries."""
  @mini_pallas.kernel
  def k(x, o):
    o[...] = x[...]

  sizes = [2, 4, 8, 16, 32]
  for n in sizes:
    x = np.ones(n)
    out = np.zeros(n)
    k(x, out)

  assert len(k._guard_cache) == len(sizes)


def test_guard_ir_matches_shape():
  """The IR stored per guard entry reflects that entry's shapes."""
  @mini_pallas.kernel
  def k(x, o):
    o[...] = x[...]

  x3, out3 = np.ones(3), np.zeros(3)
  x7, out7 = np.ones(7), np.zeros(7)

  k(x3, out3)
  k(x7, out7)

  ir3 = k.show_ir(x3, out3)
  ir7 = k.show_ir(x7, out7)

  assert "<3:" in ir3
  assert "<7:" in ir7
  assert "<7:" not in ir3
  assert "<3:" not in ir7
