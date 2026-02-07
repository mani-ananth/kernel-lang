"""Tests for mini_pallas.core — IR data structures."""

from mini_pallas.core import OpType, IRValue, IROp, KernelIR, pretty_print


def test_optype_members():
  """All expected ops exist in the enum."""
  ops = {OpType.LOAD, OpType.STORE, OpType.ADD, OpType.SUB,
         OpType.MUL, OpType.TRUEDIV, OpType.NEG, OpType.MATMUL}
  assert len(ops) == 8


def test_irvalue_repr():
  """IRValue repr returns the name."""
  v = IRValue(0, "v0")
  assert repr(v) == "v0"
  assert v.id == 0
  assert v.name == "v0"


def test_kernel_ir_new_value():
  """new_value generates incrementing IDs with correct prefix."""
  ir = KernelIR("test", ["x", "y"])
  v0 = ir.new_value()
  v1 = ir.new_value()
  v2 = ir.new_value("tmp")
  assert v0.name == "v0" and v0.id == 0
  assert v1.name == "v1" and v1.id == 1
  assert v2.name == "tmp2" and v2.id == 2


def test_kernel_ir_add_op():
  """add_op appends an op and returns the result value."""
  ir = KernelIR("test", ["x", "y"])
  v0 = ir.new_value()
  v1 = ir.new_value()
  result = ir.add_op(OpType.ADD, [v0, v1])
  assert len(ir.ops) == 1
  assert ir.ops[0].op_type == OpType.ADD
  assert ir.ops[0].result == result
  assert ir.ops[0].operands == [v0, v1]


def test_kernel_ir_add_op_no_result():
  """add_op with has_result=False returns None but still appends."""
  ir = KernelIR("test", ["x", "o"])
  v0 = ir.new_value()
  result = ir.add_op(OpType.STORE, [v0], ref_name="o", has_result=False)
  assert result is None
  assert len(ir.ops) == 1
  assert ir.ops[0].result is None
  assert ir.ops[0].ref_name == "o"


def test_pretty_print():
  """pretty_print produces expected output format."""
  ir = KernelIR("add_kernel", ["x_ref", "y_ref", "o_ref"])
  v0 = ir.add_op(OpType.LOAD, [], ref_name="x_ref")
  v1 = ir.add_op(OpType.LOAD, [], ref_name="y_ref")
  v2 = ir.add_op(OpType.ADD, [v0, v1])
  ir.add_op(OpType.STORE, [v2], ref_name="o_ref", has_result=False)

  output = pretty_print(ir)
  lines = output.strip().split("\n")
  assert lines[0] == "kernel add_kernel(x_ref, y_ref, o_ref):"
  assert "LOAD [x_ref]" in lines[1]
  assert "LOAD [y_ref]" in lines[2]
  assert "ADD" in lines[3]
  assert "STORE [o_ref]" in lines[4]
