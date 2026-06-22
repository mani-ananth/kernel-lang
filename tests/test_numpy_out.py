"""Tests for the out= NumPy lowering backend."""

import numpy as np
import pytest

import picokernel
from picokernel.lowering import lower_to_numpy
from picokernel.trace import trace_kernel


def make(fn, *shapes):
  return picokernel.KernelFunction(fn, backend="numpy")


def test_vector_add():
  @picokernel.kernel(backend="numpy")
  def k(x, y, o):
    o[...] = x[...] + y[...]

  x, y, out = np.array([1., 2., 3.]), np.array([4., 5., 6.]), np.zeros(3)
  k(x, y, out)
  np.testing.assert_array_equal(out, x + y)


def test_mul_add_chain():
  @picokernel.kernel(backend="numpy")
  def k(a, b, c, o):
    o[...] = a[...] * b[...] + c[...]

  a, b, c = np.array([1., 2., 3.]), np.array([4., 5., 6.]), np.array([1., 1., 1.])
  out = np.zeros(3)
  k(a, b, c, out)
  np.testing.assert_array_equal(out, a * b + c)


def test_negation_chain():
  @picokernel.kernel(backend="numpy")
  def k(x, y, o):
    o[...] = -(x[...] + y[...])

  x, y = np.array([1., 2., 3.]), np.array([4., 5., 6.])
  out = np.zeros(3)
  k(x, y, out)
  np.testing.assert_array_equal(out, -(x + y))


def test_matmul():
  @picokernel.kernel(backend="numpy")
  def k(a, b, o):
    o[...] = a[...] @ b[...]

  a = np.array([[1., 2.], [3., 4.]])
  b = np.array([[5., 6.], [7., 8.]])
  out = np.zeros((2, 2))
  k(a, b, out)
  np.testing.assert_array_equal(out, a @ b)


def test_load_only():
  @picokernel.kernel(backend="numpy")
  def k(x, o):
    o[...] = x[...]

  x = np.array([1., -2., 3.])
  out = np.zeros(3)
  k(x, out)
  np.testing.assert_array_equal(out, x)


def test_scalar_const():
  @picokernel.kernel(backend="numpy")
  def k(x, o):
    o[...] = x[...] * 2.0 + 1.0

  x = np.array([1., 2., 3.])
  out = np.zeros(3)
  k(x, out)
  np.testing.assert_array_equal(out, x * 2.0 + 1.0)


def test_numpy_array_const():
  @picokernel.kernel(backend="numpy")
  def k(x, o):
    o[...] = x[...] + np.array([10., 20., 30.])

  x = np.array([1., 2., 3.])
  out = np.zeros(3)
  k(x, out)
  np.testing.assert_array_equal(out, [11., 22., 33.])


def test_matmul_plus_bias():
  @picokernel.kernel(backend="numpy")
  def k(a, b, bias, o):
    o[...] = (a[...] @ b[...]) + bias[...]

  a = np.array([[1., 2.], [3., 4.]])
  b = np.eye(2)
  bias = np.array([[10., 20.], [30., 40.]])
  out = np.zeros((2, 2))
  k(a, b, bias, out)
  np.testing.assert_array_equal(out, (a @ b) + bias)


def test_lower_shows_out_param():
  @picokernel.kernel(backend="numpy")
  def k(x, y, o):
    o[...] = x[...] + y[...]

  src = k.lower()
  assert "out=" in src
  assert ".copy()" not in src
  assert "for " not in src


def test_lower_matmul_no_buf():
  """Pure matmul needs no scratch buffer — out= goes straight to store_ref."""
  def k(a, b, o):
    o[...] = a[...] @ b[...]

  ir = trace_kernel(k, [(2, 2), (2, 2), (2, 2)], [np.float32] * 3)
  src = lower_to_numpy(ir)
  assert "_buf" not in src
  assert "np.matmul(a, b, out=o)" in src
