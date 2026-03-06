"""Tests for mini_pallas.lowering — code generation."""

import numpy as np
import pytest

from mini_pallas.core import OpType, KernelIR, IROp, IRValue
from mini_pallas.lowering import lower_fused_numpy, lower_to_numpy
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


def test_lower_const_scalar():
  """CONST op lowers to a literal value."""
  ir = KernelIR("test", ["x", "o"])
  v0 = ir.add_op(OpType.LOAD, [], ref_name="x")
  v1 = ir.add_op(OpType.CONST, [], const_value=2.5)
  v2 = ir.add_op(OpType.MUL, [v0, v1])
  ir.add_op(OpType.STORE, [v2], ref_name="o", has_result=False)
  source = lower_to_numpy(ir)
  assert "v1 = 2.5" in source


def test_lower_const_with_scalar_kernel():
  """Traced kernel with scalar produces CONST in lowered code."""
  def k(x, o):
    o[...] = x[...] * 3.0
  ir = trace_kernel(k)
  source = lower_to_numpy(ir)
  assert "3.0" in source
  compile(source, "<test>", "exec")  # must be valid Python


# --- Fused lowering tests ---

def test_fused_lower_simple_add():
  """Fused lowering of simple add produces for loop."""
  def k(x, y, o):
    o[...] = x[...] + y[...]

  shapes = [(4,), (4,), (4,)]
  dtypes = [np.float64] * 3
  ir = trace_kernel(k, shapes, dtypes)
  source = lower_fused_numpy(ir)

  assert "for _i0 in range(4):" in source
  assert ".copy()" not in source
  assert "x[_i0]" in source or "x_ref[_i0]" in source  # direct ref read
  compile(source, "<test>", "exec")


def test_fused_lower_chained_ops():
  """Fused (a + b) * c produces a single for loop with scalar arithmetic."""
  def k(a, b, c, o):
    o[...] = (a[...] + b[...]) * c[...]

  shapes = [(4,), (4,), (4,), (4,)]
  dtypes = [np.float64] * 4
  ir = trace_kernel(k, shapes, dtypes)
  source = lower_fused_numpy(ir)

  assert "for _i0 in range(4):" in source
  assert "+" in source
  assert "*" in source
  assert ".copy()" not in source
  compile(source, "<test>", "exec")


def test_fused_lower_with_scalar_const():
  """Fused lowering with scalar constants emits constants before the loop."""
  def k(x, o):
    o[...] = x[...] * 2.0 + 1.0

  shapes = [(4,), (4,)]
  dtypes = [np.float64] * 2
  ir = trace_kernel(k, shapes, dtypes)
  source = lower_fused_numpy(ir)

  assert "for _i0 in range(4):" in source
  assert "2.0" in source
  assert "1.0" in source
  compile(source, "<test>", "exec")


def test_fused_lower_2d():
  """Fused lowering of 2D arrays produces nested for loops."""
  def k(x, y, o):
    o[...] = x[...] + y[...]

  shapes = [(3, 4), (3, 4), (3, 4)]
  dtypes = [np.float64] * 3
  ir = trace_kernel(k, shapes, dtypes)
  source = lower_fused_numpy(ir)

  assert "for _i0 in range(3):" in source
  assert "for _i1 in range(4):" in source
  compile(source, "<test>", "exec")


def test_fused_lower_falls_back_without_shapes():
  """Fused lowering falls back to unfused when no shape info."""
  def k(x, y, o):
    o[...] = x[...] + y[...]

  ir = trace_kernel(k)  # no shapes
  fused_source = lower_fused_numpy(ir)
  unfused_source = lower_to_numpy(ir)

  assert fused_source == unfused_source


def test_fused_lower_matmul_plus_bias():
  """Matmul is NOT fused; only the ADD with bias gets a fused loop."""
  def k(a, b, bias, o):
    o[...] = (a[...] @ b[...]) + bias[...]

  shapes = [(2, 3), (3, 4), (2, 4), (2, 4)]
  dtypes = [np.float64] * 4
  ir = trace_kernel(k, shapes, dtypes)
  source = lower_fused_numpy(ir)

  # MATMUL should still be present as unfused
  assert "@" in source
  # The ADD+bias part should be in a fused loop
  assert "for _i0" in source
  compile(source, "<test>", "exec")


def test_fused_lower_neg_chain():
  """Fused -(x + y) produces a loop with negation."""
  def k(x, y, o):
    o[...] = -(x[...] + y[...])

  shapes = [(4,), (4,), (4,)]
  dtypes = [np.float64] * 3
  ir = trace_kernel(k, shapes, dtypes)
  source = lower_fused_numpy(ir)

  assert "for _i0 in range(4):" in source
  assert "-" in source
  compile(source, "<test>", "exec")


def test_fused_lower_is_valid_python():
  """Fused lowered source compiles as valid Python."""
  def k(a, b, o):
    o[...] = (a[...] + b[...]) * a[...]

  shapes = [(8,), (8,), (8,)]
  dtypes = [np.float64] * 3
  ir = trace_kernel(k, shapes, dtypes)
  source = lower_fused_numpy(ir)
  compile(source, "<test>", "exec")
