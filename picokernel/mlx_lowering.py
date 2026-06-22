"""Lowering: convert KernelIR to executable MLX Python source."""

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
  if isinstance(value, np.ndarray):
    return f"mx.array({value.tolist()!r})"
  elif isinstance(value, np.generic):
    return repr(value.item())
  else:
    return repr(value)


def lower_to_mlx(ir: KernelIR) -> str:
  """Generate a Python/MLX function source string from the IR.

  Inputs are converted to mx.array at load time; the STORE converts the MLX
  result back to NumPy so the caller's output buffer is updated in-place.
  """
  sig = ", ".join(ir.ref_params)
  lines = [f"def {ir.name}({sig}):"]

  for op in ir.ops:
    if op.op_type == OpType.LOAD:
      lines.append(f"  {op.result} = mx.array({op.ref_name})")
    elif op.op_type == OpType.STORE:
      stored = op.operands[0]
      lines.append(f"  mx.eval({stored})")
      lines.append(f"  {op.ref_name}[...] = np.array({stored})")
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
