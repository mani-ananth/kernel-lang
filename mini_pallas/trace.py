"""Tracing system: proxy objects that record operations into a KernelIR."""

import inspect

from .core import KernelIR, IRValue, OpType


class TracerValue:
  """Proxy for a value inside a traced kernel. Arithmetic builds IR ops."""

  def __init__(self, ir: KernelIR, ir_value: IRValue):
    self._ir = ir
    self._ir_value = ir_value

  def _binop(self, other: "TracerValue", op_type: OpType) -> "TracerValue":
    result = self._ir.add_op(op_type, [self._ir_value, other._ir_value])
    return TracerValue(self._ir, result)

  def __add__(self, other):
    return self._binop(other, OpType.ADD)

  def __radd__(self, other):
    return self._binop(other, OpType.ADD)

  def __sub__(self, other):
    return self._binop(other, OpType.SUB)

  def __rsub__(self, other):
    # other - self
    return other._binop(self, OpType.SUB)

  def __mul__(self, other):
    return self._binop(other, OpType.MUL)

  def __rmul__(self, other):
    return self._binop(other, OpType.MUL)

  def __truediv__(self, other):
    return self._binop(other, OpType.TRUEDIV)

  def __rtruediv__(self, other):
    # other / self
    return other._binop(self, OpType.TRUEDIV)

  def __neg__(self):
    result = self._ir.add_op(OpType.NEG, [self._ir_value])
    return TracerValue(self._ir, result)
  
  def __matmul__(self, other):
    return self._binop(other, OpType.MATMUL)
  
  def __rmatmul__(self, other):
    return other._binop(self, OpType.MATMUL)


class TracerRef:
  """Proxy for a kernel ref parameter. Indexing with [...] emits LOAD/STORE."""

  def __init__(self, ir: KernelIR, param_name: str):
    self._ir = ir
    self._param_name = param_name

  def __getitem__(self, idx):
    if idx is not Ellipsis:
      raise NotImplementedError("Only [...] indexing is supported")
    result = self._ir.add_op(OpType.LOAD, [], ref_name=self._param_name)
    return TracerValue(self._ir, result)

  def __setitem__(self, idx, value: TracerValue):
    if idx is not Ellipsis:
      raise NotImplementedError("Only [...] indexing is supported")
    self._ir.add_op(
      OpType.STORE,
      [value._ir_value],
      ref_name=self._param_name,
      has_result=False,
    )


def trace_kernel(fn) -> KernelIR:
  """Trace a kernel function, returning the populated KernelIR."""
  sig = inspect.signature(fn)
  param_names = list(sig.parameters.keys())
  ir = KernelIR(fn.__name__, param_names)
  tracer_refs = [TracerRef(ir, name) for name in param_names]
  fn(*tracer_refs)
  return ir
