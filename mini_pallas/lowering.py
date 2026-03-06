"""Lowering: convert KernelIR to executable NumPy Python source."""

import numpy as np

from .core import KernelIR, OpType
from .passes import FusionGroup, find_fusion_groups

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


def _emit_op(op, lines, indent="  "):
  """Emit a single unfused op as a line of code."""
  if op.op_type == OpType.LOAD:
    lines.append(f"{indent}{op.result} = {op.ref_name}.copy()")
  elif op.op_type == OpType.STORE:
    lines.append(f"{indent}{op.ref_name}[...] = {op.operands[0]}")
  elif op.op_type == OpType.CONST:
    lines.append(f"{indent}{op.result} = {_format_const(op.const_value)}")
  elif op.op_type in _BINOP_SYMBOL:
    lhs, rhs = op.operands
    sym = _BINOP_SYMBOL[op.op_type]
    lines.append(f"{indent}{op.result} = {lhs} {sym} {rhs}")
  elif op.op_type == OpType.NEG:
    lines.append(f"{indent}{op.result} = -{op.operands[0]}")
  else:
    raise ValueError(f"Unsupported op: {op.op_type}")


def lower_to_numpy(ir: KernelIR) -> str:
  """Generate a Python/NumPy function source string from the IR (unfused)."""
  sig = ", ".join(ir.ref_params)
  lines = [f"def {ir.name}({sig}):"]

  for op in ir.ops:
    _emit_op(op, lines)

  return "\n".join(lines)


def _broadcast_index(idx_vars: list[str], output_shape: tuple[int, ...],
                     operand_shape: tuple[int, ...]) -> str:
  """Build an index expression for an operand inside a fused loop.

  Right-aligns operand dims to output dims.
  Size-1 dims -> index 0; real dims -> use loop variable.
  Missing leading dims -> skip.
  """
  ndim_out = len(output_shape)
  ndim_op = len(operand_shape)
  # Right-align: operand dim j aligns with output dim (ndim_out - ndim_op + j)
  indices = []
  for j in range(ndim_op):
    out_dim = ndim_out - ndim_op + j
    if operand_shape[j] == 1:
      indices.append("0")
    else:
      indices.append(idx_vars[out_dim])
  if len(indices) == 0:
    return ""
  if len(indices) == 1:
    return f"[{indices[0]}]"
  return "[" + ", ".join(indices) + "]"


def _operand_expr(val, scalar_ids: set[int], idx_vars: list[str],
                  output_shape: tuple[int, ...], ir: KernelIR,
                  array_const_names: dict[int, str]) -> str:
  """Build the expression for an operand inside the fused loop body.

  If the value was computed in this iteration (in scalar_ids), use its variable name.
  If it's an array constant, use broadcast indexing.
  Otherwise it's an external array, apply broadcast indexing.
  """
  if val.id in scalar_ids:
    return val.name

  if val.id in array_const_names:
    name = array_const_names[val.id]
    op_shape = val.shape if val.shape else ()
    idx = _broadcast_index(idx_vars, output_shape, op_shape)
    return f"{name}{idx}"

  # External array value — need broadcast indexing
  # Find the ref_name if this was a LOAD, or the variable name if materialized
  op_shape = val.shape if val.shape else output_shape
  idx = _broadcast_index(idx_vars, output_shape, op_shape)
  return f"{val.name}{idx}"


def _emit_fused_loop(group: FusionGroup, ir: KernelIR, lines: list[str]):
  """Emit a fused nested loop for a FusionGroup."""
  output_shape = group.output_shape
  ndim = len(output_shape)
  indent_base = "  "

  # Build sets of op indices in this group for quick lookup
  all_group_indices = set(group.fused_op_indices) | set(group.load_indices) | set(group.const_indices)
  all_group_indices.add(group.store_index)

  # Collect value IDs produced inside the loop (scalar context)
  scalar_ids: set[int] = set()
  for idx in group.fused_op_indices:
    op = ir.ops[idx]
    if op.result is not None:
      scalar_ids.add(op.result.id)

  # Pre-loop: declare array constants and emit scalar constants
  array_const_names: dict[int, str] = {}
  for idx in group.const_indices:
    op = ir.ops[idx]
    if isinstance(op.const_value, np.ndarray):
      cname = f"_const_{op.result.name}"
      lines.append(f"{indent_base}{cname} = {_format_const(op.const_value)}")
      array_const_names[op.result.id] = cname
    else:
      # Scalar constant: just emit it before the loop
      scalar_ids.add(op.result.id)
      lines.append(f"{indent_base}{op.result.name} = {_format_const(op.const_value)}")

  # Also mark LOAD values as scalar (they'll be read inside the loop)
  for idx in group.load_indices:
    op = ir.ops[idx]
    if op.result is not None:
      scalar_ids.add(op.result.id)

  # Index variable names
  idx_vars = [f"_i{d}" for d in range(ndim)]

  # Emit nested for loops
  for d in range(ndim):
    indent = indent_base + "  " * d
    lines.append(f"{indent}for {idx_vars[d]} in range({output_shape[d]}):")

  loop_indent = indent_base + "  " * ndim

  # Build topologically-ordered list of ops inside the loop
  # Order: loads first, then element-wise ops in original order
  loop_ops = sorted(group.load_indices + group.fused_op_indices)

  store_op = ir.ops[group.store_index]

  for op_idx in loop_ops:
    op = ir.ops[op_idx]
    if op.op_type == OpType.LOAD:
      # Direct read with broadcast index
      op_shape = op.result.shape if op.result.shape else output_shape
      idx = _broadcast_index(idx_vars, output_shape, op_shape)
      lines.append(f"{loop_indent}{op.result.name} = {op.ref_name}{idx}")
    elif op.op_type in _BINOP_SYMBOL and op.op_type != OpType.MATMUL:
      lhs, rhs = op.operands
      sym = _BINOP_SYMBOL[op.op_type]
      lhs_expr = _operand_expr(lhs, scalar_ids, idx_vars, output_shape, ir, array_const_names)
      rhs_expr = _operand_expr(rhs, scalar_ids, idx_vars, output_shape, ir, array_const_names)
      lines.append(f"{loop_indent}{op.result.name} = {lhs_expr} {sym} {rhs_expr}")
    elif op.op_type == OpType.NEG:
      operand = op.operands[0]
      expr = _operand_expr(operand, scalar_ids, idx_vars, output_shape, ir, array_const_names)
      lines.append(f"{loop_indent}{op.result.name} = -{expr}")

  # Emit store
  stored_val = store_op.operands[0]
  stored_expr = _operand_expr(stored_val, scalar_ids, idx_vars, output_shape, ir, array_const_names)
  full_idx = ", ".join(idx_vars)
  if ndim == 1:
    lines.append(f"{loop_indent}{store_op.ref_name}[{full_idx}] = {stored_expr}")
  else:
    lines.append(f"{loop_indent}{store_op.ref_name}[{full_idx}] = {stored_expr}")


def lower_fused_numpy(ir: KernelIR) -> str:
  """Generate fused Python/NumPy source: element-wise chains become nested loops."""
  groups = find_fusion_groups(ir)

  if not groups:
    return lower_to_numpy(ir)

  # Build claimed-indices set
  claimed: set[int] = set()
  group_by_store: dict[int, FusionGroup] = {}
  for g in groups:
    claimed.update(g.fused_op_indices)
    claimed.update(g.load_indices)
    claimed.update(g.const_indices)
    claimed.add(g.store_index)
    group_by_store[g.store_index] = g

  sig = ", ".join(ir.ref_params)
  lines = [f"def {ir.name}({sig}):"]

  for i, op in enumerate(ir.ops):
    if i in group_by_store:
      # Emit fused loop for this group
      _emit_fused_loop(group_by_store[i], ir, lines)
    elif i not in claimed:
      # Emit unfused op
      _emit_op(op, lines)

  return "\n".join(lines)
