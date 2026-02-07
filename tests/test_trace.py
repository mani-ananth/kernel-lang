"""Tests for mini_pallas.trace — tracing and proxy objects."""

import pytest

from mini_pallas.core import OpType
from mini_pallas.trace import trace_kernel, TracerRef, TracerValue


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
