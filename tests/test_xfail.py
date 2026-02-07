"""Tests for kernels that are expected to fail.

These document known limitations of the current tracer.
When a feature is implemented, remove the xfail marker.
"""

import numpy as np
import pytest

import mini_pallas


@pytest.mark.xfail(strict=True, reason="float has no _ir_value attribute")
def test_scalar_mul():
  """o = x * 2.0 — scalar on right side."""
  @mini_pallas.kernel
  def k(x, o):
    o[...] = x[...] * 2.0

  x = np.array([1.0, 2.0, 3.0])
  out = np.zeros(3)
  k(x, out)
  np.testing.assert_array_equal(out, [2.0, 4.0, 6.0])


@pytest.mark.xfail(strict=True, reason="float has no _ir_value attribute")
def test_scalar_add():
  """o = x + 1.0 — scalar addition."""
  @mini_pallas.kernel
  def k(x, o):
    o[...] = x[...] + 1.0

  x = np.array([1.0, 2.0, 3.0])
  out = np.zeros(3)
  k(x, out)
  np.testing.assert_array_equal(out, [2.0, 3.0, 4.0])


@pytest.mark.xfail(strict=True, reason="__pow__ not implemented on TracerValue")
def test_power():
  """o = x ** 2 — power operator."""
  @mini_pallas.kernel
  def k(x, o):
    o[...] = x[...] ** 2

  x = np.array([1.0, 2.0, 3.0])
  out = np.zeros(3)
  k(x, out)
  np.testing.assert_array_equal(out, [1.0, 4.0, 9.0])


@pytest.mark.xfail(strict=True, reason="__mod__ not implemented on TracerValue")
def test_modulo():
  """o = x % 3 — modulo operator."""
  @mini_pallas.kernel
  def k(x, o):
    o[...] = x[...] % 3

  x = np.array([1.0, 4.0, 7.0])
  out = np.zeros(3)
  k(x, out)
  np.testing.assert_array_equal(out, [1.0, 1.0, 1.0])


@pytest.mark.xfail(strict=True, reason="np.exp bypasses tracer proxy")
def test_numpy_exp():
  """o = np.exp(x) — numpy ufunc."""
  @mini_pallas.kernel
  def k(x, o):
    o[...] = np.exp(x[...])

  x = np.array([0.0, 1.0, 2.0])
  out = np.zeros(3)
  k(x, out)
  np.testing.assert_array_almost_equal(out, np.exp(x))


@pytest.mark.xfail(strict=True, reason="np.sqrt bypasses tracer proxy")
def test_numpy_sqrt():
  """o = np.sqrt(x) — numpy ufunc."""
  @mini_pallas.kernel
  def k(x, o):
    o[...] = np.sqrt(x[...])

  x = np.array([1.0, 4.0, 9.0])
  out = np.zeros(3)
  k(x, out)
  np.testing.assert_array_equal(out, [1.0, 2.0, 3.0])


@pytest.mark.xfail(strict=True, reason=".sum() not on TracerValue")
def test_reduction_sum():
  """o = x.sum() — reduction."""
  @mini_pallas.kernel
  def k(x, o):
    o[...] = x[...].sum()

  x = np.array([1.0, 2.0, 3.0, 4.0])
  out = np.zeros(1)
  k(x, out)
  np.testing.assert_array_equal(out, [10.0])


@pytest.mark.xfail(strict=True, reason=".T not on TracerValue")
def test_transpose():
  """o = x.T @ y — transpose."""
  @mini_pallas.kernel
  def k(x, y, o):
    o[...] = x[...].T @ y[...]

  x = np.array([[1.0, 2.0], [3.0, 4.0]])
  y = np.array([[1.0, 0.0], [0.0, 1.0]])
  out = np.zeros((2, 2))
  k(x, y, out)
  np.testing.assert_array_equal(out, x.T @ y)


@pytest.mark.xfail(strict=True, reason="only [...] indexing supported")
def test_non_ellipsis_index():
  """o = x[0:4] — slice indexing."""
  @mini_pallas.kernel
  def k(x, o):
    o[...] = x[0:4]

  x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
  out = np.zeros(4)
  k(x, out)
  np.testing.assert_array_equal(out, [1.0, 2.0, 3.0, 4.0])


@pytest.mark.xfail(strict=True, reason="cannot branch on TracerValue")
def test_control_flow():
  """if x > 0 — conditional branching."""
  @mini_pallas.kernel
  def k(x, o):
    val = x[...]
    if val > 0:  # This will fail: can't convert TracerValue to bool
      o[...] = val
    else:
      o[...] = -val

  x = np.array([1.0, -2.0, 3.0])
  out = np.zeros(3)
  k(x, out)
