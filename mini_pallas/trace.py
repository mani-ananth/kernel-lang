"""Tracing system: proxy objects that record operations into a KernelIR."""

import inspect
import numbers
from typing import Optional

import numpy as np

from .core import KernelIR, IRValue, OpType


def _broadcast_shapes(
  shape1: Optional[tuple[int, ...]],
  shape2: Optional[tuple[int, ...]],
) -> Optional[tuple[int, ...]]:
  """Compute broadcast result shape, or None if unknown."""
  if shape1 is None or shape2 is None:
    return None
  # Numpy-style broadcasting
  result = []
  for d1, d2 in zip(reversed(shape1), reversed(shape2)):
    if d1 == d2:
      result.append(d1)
    elif d1 == 1:
      result.append(d2)
    elif d2 == 1:
      result.append(d1)
    else:
      raise ValueError(f"Incompatible shapes for broadcasting: {shape1} vs {shape2}")
  # Handle different lengths
  longer = shape1 if len(shape1) > len(shape2) else shape2
  result.extend(reversed(longer[: len(longer) - len(result)]))
  return tuple(reversed(result))


def _matmul_shape(
  shape1: Optional[tuple[int, ...]],
  shape2: Optional[tuple[int, ...]],
) -> Optional[tuple[int, ...]]:
  """Compute matmul result shape, or None if unknown."""
  if shape1 is None or shape2 is None:
    return None
  if len(shape1) < 1 or len(shape2) < 1:
    raise ValueError(f"matmul requires at least 1D arrays: {shape1} @ {shape2}")
  # Handle 1D @ 1D -> scalar (0D)
  if len(shape1) == 1 and len(shape2) == 1:
    if shape1[0] != shape2[0]:
      raise ValueError(f"matmul dimension mismatch: {shape1} @ {shape2}")
    return ()
  # Handle 1D @ 2D -> 1D
  if len(shape1) == 1:
    if shape1[0] != shape2[-2]:
      raise ValueError(f"matmul dimension mismatch: {shape1} @ {shape2}")
    return shape2[:-2] + (shape2[-1],)
  # Handle 2D @ 1D -> 1D
  if len(shape2) == 1:
    if shape1[-1] != shape2[0]:
      raise ValueError(f"matmul dimension mismatch: {shape1} @ {shape2}")
    return shape1[:-1]
  # Handle ND @ MD -> broadcast batch dims + matmul last 2
  if shape1[-1] != shape2[-2]:
    raise ValueError(f"matmul dimension mismatch: {shape1} @ {shape2}")
  batch1, batch2 = shape1[:-2], shape2[:-2]
  batch_result = _broadcast_shapes(batch1, batch2) if batch1 or batch2 else ()
  return batch_result + (shape1[-2], shape2[-1])


def _result_dtype(
  dtype1: Optional[np.dtype],
  dtype2: Optional[np.dtype],
) -> Optional[np.dtype]:
  """Compute result dtype using numpy promotion rules."""
  if dtype1 is None or dtype2 is None:
    return None
  return np.result_type(dtype1, dtype2)


class TracerValue:
  """Proxy for a value inside a traced kernel. Arithmetic builds IR ops."""

  def __init__(
    self,
    ir: KernelIR,
    ir_value: IRValue,
    shape: Optional[tuple[int, ...]] = None,
    dtype: Optional[np.dtype] = None,
  ):
    self._ir = ir
    self._ir_value = ir_value
    self._shape = shape
    self._dtype = dtype

  def _ensure_tracer(self, other) -> "TracerValue":
    """Wrap a scalar/constant as a TracerValue with a CONST op."""
    if isinstance(other, TracerValue):
      return other
    # Accept Python numbers and numpy arrays/scalars
    if isinstance(other, (numbers.Number, np.ndarray, np.generic)):
      if isinstance(other, np.ndarray):
        shape = other.shape
        dtype = other.dtype
      elif isinstance(other, np.generic):
        shape = ()
        dtype = other.dtype
      else:
        shape = ()
        dtype = np.dtype(type(other))
      result = self._ir.add_op(OpType.CONST, [], const_value=other, shape=shape, dtype=dtype)
      return TracerValue(self._ir, result, shape, dtype)
    raise TypeError(f"Cannot convert {type(other).__name__} to TracerValue")

  def _binop(self, other, op_type: OpType) -> "TracerValue":
    other = self._ensure_tracer(other)
    if op_type == OpType.MATMUL:
      shape = _matmul_shape(self._shape, other._shape)
    else:
      shape = _broadcast_shapes(self._shape, other._shape)
    dtype = _result_dtype(self._dtype, other._dtype)
    result = self._ir.add_op(
      op_type, [self._ir_value, other._ir_value], shape=shape, dtype=dtype
    )
    return TracerValue(self._ir, result, shape, dtype)

  def _rbinop(self, other, op_type: OpType) -> "TracerValue":
    """Reverse binop: other <op> self, where other is not a TracerValue."""
    other = self._ensure_tracer(other)
    if op_type == OpType.MATMUL:
      shape = _matmul_shape(other._shape, self._shape)
    else:
      shape = _broadcast_shapes(other._shape, self._shape)
    dtype = _result_dtype(other._dtype, self._dtype)
    result = self._ir.add_op(
      op_type, [other._ir_value, self._ir_value], shape=shape, dtype=dtype
    )
    return TracerValue(self._ir, result, shape, dtype)

  def __add__(self, other):
    return self._binop(other, OpType.ADD)

  def __radd__(self, other):
    return self._binop(other, OpType.ADD)  # addition is commutative

  def __sub__(self, other):
    return self._binop(other, OpType.SUB)

  def __rsub__(self, other):
    return self._rbinop(other, OpType.SUB)  # other - self

  def __mul__(self, other):
    return self._binop(other, OpType.MUL)

  def __rmul__(self, other):
    return self._binop(other, OpType.MUL)  # multiplication is commutative

  def __truediv__(self, other):
    return self._binop(other, OpType.TRUEDIV)

  def __rtruediv__(self, other):
    return self._rbinop(other, OpType.TRUEDIV)  # other / self

  def __neg__(self):
    result = self._ir.add_op(
      OpType.NEG, [self._ir_value], shape=self._shape, dtype=self._dtype
    )
    return TracerValue(self._ir, result, self._shape, self._dtype)

  def __matmul__(self, other):
    return self._binop(other, OpType.MATMUL)

  def __rmatmul__(self, other):
    return self._rbinop(other, OpType.MATMUL)  # other @ self


class TracerRef:
  """Proxy for a kernel ref parameter. Indexing with [...] emits LOAD/STORE."""

  def __init__(
    self,
    ir: KernelIR,
    param_name: str,
    shape: Optional[tuple[int, ...]] = None,
    dtype: Optional[np.dtype] = None,
  ):
    self._ir = ir
    self._param_name = param_name
    self._shape = shape
    self._dtype = dtype

  def __getitem__(self, idx):
    if idx is not Ellipsis:
      raise NotImplementedError("Only [...] indexing is supported")
    result = self._ir.add_op(
      OpType.LOAD, [], ref_name=self._param_name, shape=self._shape, dtype=self._dtype
    )
    return TracerValue(self._ir, result, self._shape, self._dtype)

  def __setitem__(self, idx, value: TracerValue):
    if idx is not Ellipsis:
      raise NotImplementedError("Only [...] indexing is supported")
    self._ir.add_op(
      OpType.STORE,
      [value._ir_value],
      ref_name=self._param_name,
      has_result=False,
    )


def trace_kernel(
  fn,
  shapes: Optional[list[tuple[int, ...]]] = None,
  dtypes: Optional[list[np.dtype]] = None,
) -> KernelIR:
  """Trace a kernel function, returning the populated KernelIR.

  Args:
    fn: The kernel function to trace.
    shapes: Optional list of shapes for each parameter.
    dtypes: Optional list of dtypes for each parameter.
  """
  sig = inspect.signature(fn)
  param_names = list(sig.parameters.keys())
  ir = KernelIR(fn.__name__, param_names)
  tracer_refs = []
  for i, name in enumerate(param_names):
    shape = shapes[i] if shapes else None
    dtype = dtypes[i] if dtypes else None
    tracer_refs.append(TracerRef(ir, name, shape, dtype))
  fn(*tracer_refs)
  return ir
