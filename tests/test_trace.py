"""Tests for mini_pallas.trace — tracing and proxy objects."""

import numpy as np
import pytest

from mini_pallas.core import OpType
from mini_pallas.trace import trace_kernel, TracerRef, TracerValue, _broadcast_shapes, _matmul_shape


def test_trace_simple_add():
  """Tracing a simple add kernel produces correct ops."""
  def add_kernel(x_ref, y_ref, o_ref):
    x, y = x_ref[...], y_ref[...]
    o_ref[...] = x + y

  ir = trace_kernel(add_kernel)
  assert len(ir.ops) == 4
  assert ir.ops[0].op_type == OpType.LOAD
  assert ir.ops[1].op_type == OpType.LOAD
  assert ir.ops[2].op_type == OpType.ADD
  assert ir.ops[3].op_type == OpType.STORE


def test_trace_param_names():
  """ref_params match the function signature."""
  def my_kernel(alpha, beta, gamma):
    a = alpha[...]
    gamma[...] = a

  ir = trace_kernel(my_kernel)
  assert ir.ref_params == ["alpha", "beta", "gamma"]
  assert ir.name == "my_kernel"


def test_trace_binop_add():
  def k(x, y, o):
    o[...] = x[...] + y[...]
  ir = trace_kernel(k)
  assert any(op.op_type == OpType.ADD for op in ir.ops)


def test_trace_binop_sub():
  def k(x, y, o):
    o[...] = x[...] - y[...]
  ir = trace_kernel(k)
  assert any(op.op_type == OpType.SUB for op in ir.ops)


def test_trace_binop_mul():
  def k(x, y, o):
    o[...] = x[...] * y[...]
  ir = trace_kernel(k)
  assert any(op.op_type == OpType.MUL for op in ir.ops)


def test_trace_binop_truediv():
  def k(x, y, o):
    o[...] = x[...] / y[...]
  ir = trace_kernel(k)
  assert any(op.op_type == OpType.TRUEDIV for op in ir.ops)


def test_trace_binop_matmul():
  def k(x, y, o):
    o[...] = x[...] @ y[...]
  ir = trace_kernel(k)
  assert any(op.op_type == OpType.MATMUL for op in ir.ops)


def test_trace_neg():
  """Unary negation produces a NEG op."""
  def k(x, o):
    o[...] = -x[...]
  ir = trace_kernel(k)
  assert any(op.op_type == OpType.NEG for op in ir.ops)


def test_trace_chained_ops():
  """Chained operations produce correct sequence."""
  def k(a, b, c, o):
    o[...] = (a[...] + b[...]) * c[...]
  ir = trace_kernel(k)
  op_types = [op.op_type for op in ir.ops]
  # 3 LOADs, then ADD, then MUL, then STORE
  assert op_types.count(OpType.LOAD) == 3
  assert OpType.ADD in op_types
  assert OpType.MUL in op_types
  assert op_types[-1] == OpType.STORE


def test_tracer_ref_non_ellipsis():
  """Non-ellipsis indexing raises NotImplementedError."""
  from mini_pallas.core import KernelIR
  ir = KernelIR("test", ["x"])
  ref = TracerRef(ir, "x")
  with pytest.raises(NotImplementedError):
    _ = ref[0]
  with pytest.raises(NotImplementedError):
    _ = ref[0:4]


# --- Shape tracking tests ---

def test_trace_with_shapes():
  """Tracing with shapes propagates them through ops."""
  def k(x, y, o):
    o[...] = x[...] + y[...]
  shapes = [(4,), (4,), (4,)]
  dtypes = [np.float64, np.float64, np.float64]
  ir = trace_kernel(k, shapes, dtypes)
  # LOAD should have shape
  assert ir.ops[0].result.shape == (4,)
  assert ir.ops[0].result.dtype == np.float64
  # ADD should have shape
  assert ir.ops[2].result.shape == (4,)


def test_trace_matmul_shapes():
  """Matmul shape inference produces correct result shapes."""
  def k(a, b, o):
    o[...] = a[...] @ b[...]
  shapes = [(3, 4), (4, 5), (3, 5)]
  dtypes = [np.float32, np.float32, np.float32]
  ir = trace_kernel(k, shapes, dtypes)
  # Find MATMUL op
  matmul_op = next(op for op in ir.ops if op.op_type == OpType.MATMUL)
  assert matmul_op.result.shape == (3, 5)


def test_broadcast_shapes_same():
  """Identical shapes broadcast to themselves."""
  assert _broadcast_shapes((3, 4), (3, 4)) == (3, 4)


def test_broadcast_shapes_scalar():
  """Scalar broadcasts to any shape."""
  assert _broadcast_shapes((3, 4), ()) == (3, 4)
  assert _broadcast_shapes((), (3, 4)) == (3, 4)


def test_broadcast_shapes_one_dim():
  """Broadcasting with 1-sized dimensions."""
  assert _broadcast_shapes((3, 1), (1, 4)) == (3, 4)
  assert _broadcast_shapes((1, 4), (3, 1)) == (3, 4)


def test_broadcast_shapes_different_ndim():
  """Broadcasting with different number of dimensions."""
  assert _broadcast_shapes((4,), (3, 4)) == (3, 4)
  assert _broadcast_shapes((3, 4), (4,)) == (3, 4)


def test_broadcast_shapes_incompatible():
  """Incompatible shapes raise ValueError."""
  with pytest.raises(ValueError, match="Incompatible shapes"):
    _broadcast_shapes((3,), (4,))


def test_matmul_shape_2d():
  """2D @ 2D matmul shape."""
  assert _matmul_shape((3, 4), (4, 5)) == (3, 5)


def test_matmul_shape_1d_1d():
  """1D @ 1D is a dot product (scalar)."""
  assert _matmul_shape((4,), (4,)) == ()


def test_matmul_shape_2d_1d():
  """2D @ 1D is a matrix-vector product."""
  assert _matmul_shape((3, 4), (4,)) == (3,)


def test_matmul_shape_1d_2d():
  """1D @ 2D is a vector-matrix product."""
  assert _matmul_shape((4,), (4, 5)) == (5,)


def test_matmul_shape_mismatch():
  """Mismatched inner dimensions raise ValueError."""
  with pytest.raises(ValueError, match="dimension mismatch"):
    _matmul_shape((3, 4), (5, 6))
