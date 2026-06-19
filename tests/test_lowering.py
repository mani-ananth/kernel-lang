"""Tests for mini_pallas.lowering — code generation."""

import numpy as np
import pytest

from mini_pallas.core import IROp, IRValue, KernelIR, OpType
from mini_pallas.lowering import lower_to_numpy
from mini_pallas.trace import trace_kernel


def test_lower_add_kernel():
  """Lowered add kernel uses out= and no copies."""
  def k(x, y, o):
    o[...] = x[...] + y[...]

  ir = trace_kernel(k)
  source = lower_to_numpy(ir)

  assert "def k(x, y, o):" in source
  assert "np.add(" in source
  assert "out=" in source
  assert ".copy()" not in source
  compile(source, "<test>", "exec")


def test_lower_binop_ufuncs():
  """Each binary op lowers to the correct np ufunc."""
  cases = {
    OpType.ADD: "np.add",
    OpType.SUB: "np.subtract",
    OpType.MUL: "np.multiply",
    OpType.TRUEDIV: "np.true_divide",
    OpType.MATMUL: "np.matmul",
  }
  for op_type, expected_fn in cases.items():
    ir = KernelIR("test", ["x", "y", "o"])
    v0 = ir.add_op(OpType.LOAD, [], ref_name="x")
    v1 = ir.add_op(OpType.LOAD, [], ref_name="y")
    v2 = ir.add_op(op_type, [v0, v1])
    ir.add_op(OpType.STORE, [v2], ref_name="o", has_result=False)
    source = lower_to_numpy(ir)
    assert expected_fn in source, f"Expected '{expected_fn}' for {op_type}"


def test_lower_neg():
  """NEG op lowers to np.negative."""
  def k(x, o):
    o[...] = -x[...]

  ir = trace_kernel(k)
  source = lower_to_numpy(ir)
  assert "np.negative(" in source
  compile(source, "<test>", "exec")


def test_lower_is_valid_python():
  """Lowered source is valid Python."""
  def k(a, b, o):
    o[...] = (a[...] + b[...]) * a[...]

  ir = trace_kernel(k)
  compile(lower_to_numpy(ir), "<test>", "exec")


def test_lower_const_scalar():
  """CONST op lowers to a literal value."""
  ir = KernelIR("test", ["x", "o"])
  v0 = ir.add_op(OpType.LOAD, [], ref_name="x")
  v1 = ir.add_op(OpType.CONST, [], const_value=2.5)
  v2 = ir.add_op(OpType.MUL, [v0, v1])
  ir.add_op(OpType.STORE, [v2], ref_name="o", has_result=False)
  source = lower_to_numpy(ir)
  assert "2.5" in source


def test_lower_matmul_no_scratch():
  """Pure matmul writes directly to output — no _buf needed."""
  def k(a, b, o):
    o[...] = a[...] @ b[...]

  ir = trace_kernel(k)
  source = lower_to_numpy(ir)
  assert "_buf" not in source
  assert "np.matmul(" in source
  assert "out=o" in source


def test_lower_chain_uses_scratch():
  """a*b+c needs one scratch buffer for the intermediate multiply."""
  def k(a, b, c, o):
    o[...] = a[...] * b[...] + c[...]

  ir = trace_kernel(k)
  source = lower_to_numpy(ir)
  assert "_buf = np.empty_like(o)" in source
  assert "np.multiply(a, b, out=_buf)" in source
  assert "np.add(" in source
  assert "out=o" in source


def test_lower_load_only():
  """Load-only kernel emits a direct assignment, no ufunc."""
  def k(x, o):
    o[...] = x[...]

  ir = trace_kernel(k)
  source = lower_to_numpy(ir)
  assert "o[...] = x" in source
  assert "np." not in source


def test_lower_no_for_loops():
  """lower_to_numpy never emits Python for-loops."""
  def k(a, b, c, o):
    o[...] = (a[...] + b[...]) * c[...]

  ir = trace_kernel(k)
  assert "for " not in lower_to_numpy(ir)


def test_lower_unsupported_op():
  """Unknown op type raises an error."""
  ir = KernelIR("test", ["x", "o"])
  v0 = ir.add_op(OpType.LOAD, [], ref_name="x")

  class FakeOpType:
    name = "FAKE"

  fake_result = IRValue(99, "v99")
  ir.ops.append(IROp(FakeOpType(), fake_result, [v0]))
  ir.add_op(OpType.STORE, [fake_result], ref_name="o", has_result=False)
  with pytest.raises((ValueError, KeyError)):
    lower_to_numpy(ir)
