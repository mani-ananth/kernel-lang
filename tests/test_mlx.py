"""MLX backend tests — skipped if mlx is not installed."""

import numpy as np
import pytest

mlx = pytest.importorskip("mlx.core")

import picokernel
from picokernel.mlx_lowering import lower_to_mlx
from picokernel.trace import trace_kernel


@pytest.fixture
def add_kernel():
  @picokernel.kernel(backend="mlx")
  def k(x_ref, y_ref, o_ref):
    x, y = x_ref[...], y_ref[...]
    o_ref[...] = x + y
  return k


def test_vector_add(add_kernel):
  x = np.array([1.0, 2.0, 3.0, 4.0])
  y = np.array([5.0, 6.0, 7.0, 8.0])
  out = np.zeros(4)
  add_kernel(x, y, out)
  np.testing.assert_array_almost_equal(out, x + y)


def test_elementwise_mul():
  @picokernel.kernel(backend="mlx")
  def k(x, y, o):
    o[...] = x[...] * y[...]

  x = np.array([2.0, 3.0, 4.0])
  y = np.array([5.0, 6.0, 7.0])
  out = np.zeros(3)
  k(x, y, out)
  np.testing.assert_array_almost_equal(out, x * y)


def test_negate():
  @picokernel.kernel(backend="mlx")
  def k(x, o):
    o[...] = -x[...]

  x = np.array([1.0, -2.0, 3.0])
  out = np.zeros(3)
  k(x, out)
  np.testing.assert_array_almost_equal(out, -x)


def test_chained_ops():
  @picokernel.kernel(backend="mlx")
  def k(a, b, c, o):
    o[...] = (a[...] + b[...]) * c[...]

  a = np.array([1.0, 2.0])
  b = np.array([3.0, 4.0])
  c = np.array([2.0, 3.0])
  out = np.zeros(2)
  k(a, b, c, out)
  np.testing.assert_array_almost_equal(out, (a + b) * c)


def test_matrix_multiply():
  @picokernel.kernel(backend="mlx")
  def k(a, b, o):
    o[...] = a[...] @ b[...]

  a = np.array([[1.0, 2.0], [3.0, 4.0]])
  b = np.array([[5.0, 6.0], [7.0, 8.0]])
  out = np.zeros((2, 2))
  k(a, b, out)
  np.testing.assert_array_almost_equal(out, a @ b)


def test_scalar_const():
  @picokernel.kernel(backend="mlx")
  def k(x, o):
    o[...] = x[...] * 2.0 + 1.0

  x = np.array([1.0, 2.0, 3.0])
  out = np.zeros(3)
  k(x, out)
  np.testing.assert_array_almost_equal(out, x * 2.0 + 1.0)


def test_numpy_array_const():
  @picokernel.kernel(backend="mlx")
  def k(x, o):
    o[...] = x[...] + np.array([10.0, 20.0, 30.0])

  x = np.array([1.0, 2.0, 3.0])
  out = np.zeros(3)
  k(x, out)
  np.testing.assert_array_almost_equal(out, [11.0, 22.0, 33.0])


def test_retrace_on_shape_change():
  @picokernel.kernel(backend="mlx")
  def k(x, o):
    o[...] = x[...] * 2.0

  x1 = np.array([1.0, 2.0, 3.0])
  out1 = np.zeros(3)
  k(x1, out1)
  np.testing.assert_array_almost_equal(out1, x1 * 2.0)

  x2 = np.ones(5)
  out2 = np.zeros(5)
  k(x2, out2)
  np.testing.assert_array_almost_equal(out2, x2 * 2.0)


def test_lower_returns_mlx_source():
  @picokernel.kernel(backend="mlx")
  def k(x, y, o):
    o[...] = x[...] + y[...]

  source = k.lower()
  assert "mx.array" in source
  assert "mx.eval" in source
  assert "np.array" in source


def test_lower_to_mlx_directly():
  def k(x, y, o):
    o[...] = x[...] + y[...]

  ir = trace_kernel(k)
  source = lower_to_mlx(ir)
  assert "mx.array" in source
  assert "def k(" in source
