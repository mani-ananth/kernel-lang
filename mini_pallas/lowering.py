"""Lowering: convert KernelIR to executable NumPy Python source."""

import numpy as np

from .core import KernelIR, OpType

_BINOP_SYMBOL = {
  OpType.ADD: "+",
  OpType.SUB: "-",
  OpType.MUL: "*",
  OpType.TRUEDIV: "/",
  OpType.MATMUL: "@",
}


def _format_const(value) -> str:
  """Format a constant value for code generation."""
  if isinstance(value, np.ndarray):
    # For numpy arrays, generate np.array(...) call
    return f"np.array({value.tolist()!r}, dtype=np.{value.dtype})"
  elif isinstance(value, np.generic):
    # For numpy scalars, convert to Python scalar
    return repr(value.item())
  else:
    return repr(value)


def lower_to_numpy(ir: KernelIR) -> str:
  """Generate a Python/NumPy function source string from the IR."""
  sig = ", ".join(ir.ref_params)
  lines = [f"def {ir.name}({sig}):"]

  for op in ir.ops:
    if op.op_type == OpType.LOAD:
      lines.append(f"  {op.result} = {op.ref_name}.copy()")
    elif op.op_type == OpType.STORE:
      lines.append(f"  {op.ref_name}[...] = {op.operands[0]}")
    elif op.op_type == OpType.CONST:
      lines.append(f"  {op.result} = {_format_const(op.const_value)}")
    elif op.op_type in _BINOP_SYMBOL:
      lhs, rhs = op.operands
      sym = _BINOP_SYMBOL[op.op_type]
      lines.append(f"  {op.result} = {lhs} {sym} {rhs}")
    elif op.op_type == OpType.NEG:
      lines.append(f"  {op.result} = -{op.operands[0]}")
    else:
      raise ValueError(f"Unsupported op: {op.op_type}")

  return "\n".join(lines)
