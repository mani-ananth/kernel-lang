"""IR definitions for mini_pallas: OpType, IRValue, IROp, KernelIR."""

from enum import Enum, auto
from typing import Any, Optional


class OpType(Enum):
  LOAD = auto()
  STORE = auto()
  CONST = auto()
  ADD = auto()
  SUB = auto()
  MUL = auto()
  TRUEDIV = auto()
  NEG = auto()
  MATMUL = auto()


class IRValue:
  """SSA value with a unique ID."""

  def __init__(self, id: int, name: str):
    self.id = id
    self.name = name

  def __repr__(self):
    return self.name


class IROp:
  """Single IR operation."""

  def __init__(
    self,
    op_type: OpType,
    result: Optional[IRValue],
    operands: list[IRValue],
    ref_name: Optional[str] = None,
    const_value: Any = None,
  ):
    self.op_type = op_type
    self.result = result
    self.operands = operands
    self.ref_name = ref_name  # used by LOAD/STORE to track which ref
    self.const_value = const_value  # used by CONST to store literal value


class KernelIR:
  """Full kernel IR: list of ops plus a value factory."""

  def __init__(self, name: str, ref_params: list[str]):
    self.name = name
    self.ref_params = ref_params
    self.ops: list[IROp] = []
    self._next_id = 0

  def new_value(self, prefix: str = "v") -> IRValue:
    val = IRValue(self._next_id, f"{prefix}{self._next_id}")
    self._next_id += 1
    return val

  def add_op(
    self,
    op_type: OpType,
    operands: list[IRValue],
    ref_name: Optional[str] = None,
    has_result: bool = True,
    const_value: Any = None,
  ) -> Optional[IRValue]:
    result = self.new_value() if has_result else None
    op = IROp(op_type, result, operands, ref_name, const_value)
    self.ops.append(op)
    return result


def pretty_print(ir: KernelIR) -> str:
  """Human-readable IR dump."""
  lines = [f"kernel {ir.name}({', '.join(ir.ref_params)}):"]
  for op in ir.ops:
    operand_str = ", ".join(str(o) for o in op.operands)
    ref_part = f" [{op.ref_name}]" if op.ref_name else ""
    if op.op_type == OpType.CONST:
      lines.append(f"  {op.result} = CONST({op.const_value!r})")
    elif op.result is not None:
      lines.append(f"  {op.result} = {op.op_type.name}{ref_part}({operand_str})")
    else:
      lines.append(f"  {op.op_type.name}{ref_part}({operand_str})")
  return "\n".join(lines)
