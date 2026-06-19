"""Lowering: convert KernelIR to executable NumPy Python source."""

import numpy as np

from .core import KernelIR, OpType

_NUMPY_UFUNC = {
  OpType.ADD: "np.add",
  OpType.SUB: "np.subtract",
  OpType.MUL: "np.multiply",
  OpType.TRUEDIV: "np.true_divide",
  OpType.MATMUL: "np.matmul",
  OpType.NEG: "np.negative",
}


def _format_const(value) -> str:
  if isinstance(value, np.ndarray):
    return f"np.array({value.tolist()!r}, dtype=np.{value.dtype})"
  elif isinstance(value, np.generic):
    return repr(value.item())
  else:
    return repr(value)


def lower_to_numpy(ir: KernelIR) -> str:
  """Generate NumPy source using ufunc out= to eliminate intermediate allocations.

  - LOADs are aliased directly to their ref (no .copy())
  - The op feeding STORE writes via out=store_ref (zero-copy into output)
  - Single-use intermediates write via out=_buf (one pre-allocated scratch)
  - Multi-use intermediates allocate normally (must be kept alive)
  """
  use_count: dict[int, int] = {}
  for op in ir.ops:
    for v in op.operands:
      use_count[v.id] = use_count.get(v.id, 0) + 1

  alias: dict[int, str] = {
    op.result.id: op.ref_name
    for op in ir.ops if op.op_type == OpType.LOAD
  }

  store_op = next(op for op in ir.ops if op.op_type == OpType.STORE)
  store_ref = store_op.ref_name
  store_val_id = store_op.operands[0].id
  store_val_is_alias = store_val_id in alias

  def resolve(v) -> str:
    return alias.get(v.id, v.name)

  def out_for(result_id: int) -> str | None:
    if result_id == store_val_id and not store_val_is_alias:
      return store_ref
    if use_count.get(result_id, 0) == 1:
      return "_buf"
    return None

  needs_buf = any(
    op.result
    and op.result.id != store_val_id
    and op.result.id not in alias
    and use_count.get(op.result.id, 0) == 1
    and op.op_type not in (OpType.LOAD, OpType.CONST)
    for op in ir.ops
  )

  sig = ", ".join(ir.ref_params)
  lines = [f"def {ir.name}({sig}):"]
  if needs_buf:
    lines.append(f"  _buf = np.empty_like({store_ref})")

  for op in ir.ops:
    if op.op_type in (OpType.LOAD, OpType.STORE):
      continue

    if op.op_type == OpType.CONST:
      lines.append(f"  {op.result.name} = {_format_const(op.const_value)}")
      continue

    fn = _NUMPY_UFUNC[op.op_type]
    args = ", ".join(resolve(v) for v in op.operands)
    out = out_for(op.result.id)

    if out:
      is_final = op.result.id == store_val_id
      call = f"{fn}({args}, out={out})"
      lines.append(f"  {call}" if is_final else f"  {op.result.name} = {call}")
    else:
      lines.append(f"  {op.result.name} = {fn}({args})")

  if store_val_is_alias:
    lines.append(f"  {store_ref}[...] = {alias[store_val_id]}")

  return "\n".join(lines)
