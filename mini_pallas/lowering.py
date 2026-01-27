"""Lowering: convert KernelIR to executable NumPy Python source."""

from .core import KernelIR, OpType

_BINOP_SYMBOL = {
  OpType.ADD: "+",
  OpType.SUB: "-",
  OpType.MUL: "*",
  OpType.TRUEDIV: "/",
  OpType.MATMUL: "@",
}


def lower_to_numpy(ir: KernelIR) -> str:
  """Generate a Python/NumPy function source string from the IR."""
  sig = ", ".join(ir.ref_params)
  lines = [f"def {ir.name}({sig}):"]

  for op in ir.ops:
    if op.op_type == OpType.LOAD:
      lines.append(f"  {op.result} = {op.ref_name}.copy()")
    elif op.op_type == OpType.STORE:
      lines.append(f"  {op.ref_name}[...] = {op.operands[0]}")
    elif op.op_type in _BINOP_SYMBOL:
      lhs, rhs = op.operands
      sym = _BINOP_SYMBOL[op.op_type]
      lines.append(f"  {op.result} = {lhs} {sym} {rhs}")
    elif op.op_type == OpType.NEG:
      lines.append(f"  {op.result} = -{op.operands[0]}")
    else:
      raise ValueError(f"Unsupported op: {op.op_type}")

  return "\n".join(lines)
