"""Integration tests — end-to-end kernel execution."""

import numpy as np

import mini_pallas


def test_vector_add():
  """Vector addition produces correct result."""
  @mini_pallas.kernel
  def add_kernel(x_ref, y_ref, o_ref):
    x, y = x_ref[...], y_ref[...]
    o_ref[...] = x + y

  x = np.array([1.0, 2.0, 3.0, 4.0])
  y = np.array([5.0, 6.0, 7.0, 8.0])
  out = np.zeros(4)
  add_kernel(x, y, out)
  expected = np.array([6.0, 8.0, 10.0, 12.0])
  np.testing.assert_array_equal(out, expected)


def test_matrix_multiply():
  """Matrix multiplication produces correct result."""
  @mini_pallas.kernel
  def matmul_kernel(a_ref, b_ref, o_ref):
    a, b = a_ref[...], b_ref[...]
    o_ref[...] = a @ b

  a = np.array([[1.0, 2.0], [3.0, 4.0]])
  b = np.array([[5.0, 6.0], [7.0, 8.0]])
  out = np.zeros((2, 2))
  matmul_kernel(a, b, out)
  expected = a @ b
  np.testing.assert_array_equal(out, expected)


def test_elementwise_sub():
  """Subtraction end-to-end."""
  @mini_pallas.kernel
  def k(x, y, o):
    o[...] = x[...] - y[...]

  x = np.array([10.0, 20.0, 30.0])
  y = np.array([1.0, 2.0, 3.0])
  out = np.zeros(3)
  k(x, y, out)
  np.testing.assert_array_equal(out, [9.0, 18.0, 27.0])


def test_elementwise_mul():
  """Multiplication end-to-end."""
  @mini_pallas.kernel
  def k(x, y, o):
    o[...] = x[...] * y[...]

  x = np.array([2.0, 3.0, 4.0])
  y = np.array([5.0, 6.0, 7.0])
  out = np.zeros(3)
  k(x, y, out)
  np.testing.assert_array_equal(out, [10.0, 18.0, 28.0])


def test_elementwise_div():
  """Division end-to-end."""
  @mini_pallas.kernel
  def k(x, y, o):
    o[...] = x[...] / y[...]

  x = np.array([10.0, 20.0, 30.0])
  y = np.array([2.0, 4.0, 5.0])
  out = np.zeros(3)
  k(x, y, out)
  np.testing.assert_array_equal(out, [5.0, 5.0, 6.0])


def test_negate():
  """Negation with 2-param kernel."""
  @mini_pallas.kernel
  def k(x, o):
    o[...] = -x[...]

  x = np.array([1.0, -2.0, 3.0])
  out = np.zeros(3)
  k(x, out)
  np.testing.assert_array_equal(out, [-1.0, 2.0, -3.0])


def test_chained_ops():
  """Chained (a + b) * c end-to-end."""
  @mini_pallas.kernel
  def k(a, b, c, o):
    o[...] = (a[...] + b[...]) * c[...]

  a = np.array([1.0, 2.0])
  b = np.array([3.0, 4.0])
  c = np.array([2.0, 3.0])
  out = np.zeros(2)
  k(a, b, c, out)
  expected = (a + b) * c
  np.testing.assert_array_equal(out, expected)


def test_dot_add():
  """(a @ b) + bias end-to-end."""
  @mini_pallas.kernel
  def k(a, b, bias, o):
    o[...] = (a[...] @ b[...]) + bias[...]

  a = np.array([[1.0, 2.0], [3.0, 4.0]])
  b = np.array([[1.0, 0.0], [0.0, 1.0]])
  bias = np.array([[10.0, 20.0], [30.0, 40.0]])
  out = np.zeros((2, 2))
  k(a, b, bias, out)
  expected = (a @ b) + bias
  np.testing.assert_array_equal(out, expected)


def test_show_ir_returns_string():
  """.show_ir() returns a non-empty string."""
  @mini_pallas.kernel
  def k(x, o):
    o[...] = x[...]

  ir_str = k.show_ir()
  assert isinstance(ir_str, str)
  assert len(ir_str) > 0
  assert "kernel k" in ir_str


def test_lower_returns_string():
  """.lower() returns a non-empty string."""
  @mini_pallas.kernel
  def k(x, o):
    o[...] = x[...]

  source = k.lower()
  assert isinstance(source, str)
  assert "def k(" in source


def test_scalar_mul():
  """Scalar multiplication: x * 2.0."""
  @mini_pallas.kernel
  def k(x, o):
    o[...] = x[...] * 2.0

  x = np.array([1.0, 2.0, 3.0])
  out = np.zeros(3)
  k(x, out)
  np.testing.assert_array_equal(out, [2.0, 4.0, 6.0])


def test_scalar_add():
  """Scalar addition: x + 1.0."""
  @mini_pallas.kernel
  def k(x, o):
    o[...] = x[...] + 1.0

  x = np.array([1.0, 2.0, 3.0])
  out = np.zeros(3)
  k(x, out)
  np.testing.assert_array_equal(out, [2.0, 3.0, 4.0])


def test_scalar_on_left():
  """Scalar on left side: 2.0 * x."""
  @mini_pallas.kernel
  def k(x, o):
    o[...] = 2.0 * x[...]

  x = np.array([1.0, 2.0, 3.0])
  out = np.zeros(3)
  k(x, out)
  np.testing.assert_array_equal(out, [2.0, 4.0, 6.0])


def test_scalar_sub_left():
  """Scalar on left of subtraction: 10.0 - x."""
  @mini_pallas.kernel
  def k(x, o):
    o[...] = 10.0 - x[...]

  x = np.array([1.0, 2.0, 3.0])
  out = np.zeros(3)
  k(x, out)
  np.testing.assert_array_equal(out, [9.0, 8.0, 7.0])


def test_scalar_div():
  """Scalar division: x / 2.0."""
  @mini_pallas.kernel
  def k(x, o):
    o[...] = x[...] / 2.0

  x = np.array([2.0, 4.0, 6.0])
  out = np.zeros(3)
  k(x, out)
  np.testing.assert_array_equal(out, [1.0, 2.0, 3.0])


def test_numpy_array_const():
  """NumPy array as constant: x + np.array([1, 2, 3])."""
  @mini_pallas.kernel
  def k(x, o):
    o[...] = x[...] + np.array([10.0, 20.0, 30.0])

  x = np.array([1.0, 2.0, 3.0])
  out = np.zeros(3)
  k(x, out)
  np.testing.assert_array_equal(out, [11.0, 22.0, 33.0])


def test_affine_transform():
  """Affine transform: x * scale + bias with scalars."""
  @mini_pallas.kernel
  def k(x, o):
    o[...] = x[...] * 2.0 + 1.0

  x = np.array([1.0, 2.0, 3.0])
  out = np.zeros(3)
  k(x, out)
  np.testing.assert_array_equal(out, [3.0, 5.0, 7.0])


def test_show_ir_with_shapes():
  """show_ir with arrays shows shape information."""
  @mini_pallas.kernel
  def k(x, y, o):
    o[...] = x[...] + y[...]

  x = np.array([1.0, 2.0, 3.0])
  y = np.array([4.0, 5.0, 6.0])
  out = np.zeros(3)
  ir_str = k.show_ir(x, y, out)
  # Should contain shape annotations
  assert "<3:float64>" in ir_str


def test_show_ir_matmul_shapes():
  """show_ir shows correct matmul result shape."""
  @mini_pallas.kernel
  def k(a, b, o):
    o[...] = a[...] @ b[...]

  a = np.zeros((3, 4))
  b = np.zeros((4, 5))
  out = np.zeros((3, 5))
  ir_str = k.show_ir(a, b, out)
  # Should show 3,5 shape for matmul result
  assert "<3,5:" in ir_str


def test_kernel_retraces_on_shape_change():
  """Kernel retraces when input shapes change."""
  @mini_pallas.kernel
  def k(x, o):
    o[...] = x[...] * 2.0

  # First call with shape (3,)
  x1 = np.array([1.0, 2.0, 3.0])
  out1 = np.zeros(3)
  k(x1, out1)
  ir1 = k.show_ir(x1, out1)

  # Second call with shape (5,)
  x2 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
  out2 = np.zeros(5)
  k(x2, out2)
  ir2 = k.show_ir(x2, out2)

  # Should have different shapes in IR
  assert "<3:" in ir1
  assert "<5:" in ir2
