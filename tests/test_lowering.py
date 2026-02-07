"""Tests for mini_pallas.lowering — code generation."""

import pytest

from mini_pallas.core import OpType, KernelIR, IROp, IRValue
from mini_pallas.lowering import lower_to_numpy
from mini_pallas.trace import trace_kernel


def test_lower_add_kernel():
  """Lowered add kernel produces expected source."""
  def add_kernel(x_ref, y_ref, o_ref):
    x, y = x_ref[...], y_ref[...]
    o_ref[...] = x + y

  ir = trace_kernel(add_kernel)
  source = lower_to_numpy(ir)

  assert "def add_kernel(x_ref, y_ref, o_ref):" in source
  assert "x_ref.copy()" in source
  assert "y_ref.copy()" in source
  assert "+" in source
  assert "o_ref[...] =" in source


def test_lower_binop_symbols():
  """Each binary op lowers to the correct symbol."""
  symbols = {
    OpType.ADD: "+",
    OpType.SUB: "-",
    OpType.MUL: "*",
    OpType.TRUEDIV: "/",
    OpType.MATMUL: "@",
  }
  for op_type, expected_sym in symbols.items():
    ir = KernelIR("test", ["x", "y", "o"])
    v0 = ir.add_op(OpType.LOAD, [], ref_name="x")
    v1 = ir.add_op(OpType.LOAD, [], ref_name="y")
    v2 = ir.add_op(op_type, [v0, v1])
    ir.add_op(OpType.STORE, [v2], ref_name="o", has_result=False)
    source = lower_to_numpy(ir)
    assert f" {expected_sym} " in source, f"Expected '{expected_sym}' for {op_type}"


def test_lower_neg():
  """NEG op lowers to unary minus."""
  def k(x, o):
    o[...] = -x[...]
  ir = trace_kernel(k)
  source = lower_to_numpy(ir)
  assert "= -v" in source


def test_lower_is_valid_python():
  """Lowered source is valid Python that compiles."""
  def k(a, b, o):
    o[...] = (a[...] + b[...]) * a[...]
  ir = trace_kernel(k)
  source = lower_to_numpy(ir)
  # Should not raise
  compile(source, "<test>", "exec")


def test_lower_unsupported_op():
  """Unknown op type raises ValueError."""
  # Create a fake op with an invalid type by patching
  ir = KernelIR("test", ["x", "o"])
  v0 = ir.add_op(OpType.LOAD, [], ref_name="x")

  # Manually create an op with a type that's not in the lowering table
  # We'll use NEG but then patch it to something unexpected
  class FakeOpType:
    name = "FAKE"
  fake_op = IROp(FakeOpType(), IRValue(99, "v99"), [v0])
  ir.ops.append(fake_op)

  with pytest.raises(ValueError, match="Unsupported op"):
    lower_to_numpy(ir)
