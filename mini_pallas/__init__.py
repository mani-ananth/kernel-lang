"""mini_pallas — a minimal Pallas-like kernel language."""

from .core import pretty_print
from .lowering import lower_fused_numpy, lower_to_numpy
from .runtime import compile_fused_numpy, compile_numpy
from .trace import trace_kernel


class KernelFunction:
  """Wraps a user-defined kernel function with trace/lower/compile/run.

  Guard-based retracing: each unique (shapes, dtypes) signature gets its own
  cached (KernelIR, compiled_fn) entry, mirroring JAX/Dynamo's approach.
  """

  def __init__(self, fn):
    self._fn = fn
    # Guard cache: (shapes_tuple, dtypes_tuple) -> (KernelIR, compiled_fn)
    self._guard_cache: dict[tuple, tuple] = {}

  def _guard_key(self, arrays) -> tuple:
    """Build a hashable guard key from array shapes and dtypes."""
    shapes = tuple(arr.shape for arr in arrays)
    dtypes = tuple(arr.dtype for arr in arrays)
    return (shapes, dtypes)

  def _get_or_trace(self, arrays) -> tuple:
    """Return (ir, compiled_fn), tracing + compiling only on guard miss."""
    key = self._guard_key(arrays)
    if key not in self._guard_cache:
      shapes = [arr.shape for arr in arrays]
      dtypes = [arr.dtype for arr in arrays]
      ir = trace_kernel(self._fn, shapes, dtypes)
      fn = compile_fused_numpy(ir)
      self._guard_cache[key] = (ir, fn)
    return self._guard_cache[key]

  @property
  def ir(self):
    """Return the most recently cached IR, or trace without shape info."""
    if self._guard_cache:
      return next(iter(self._guard_cache.values()))[0]
    return trace_kernel(self._fn)

  def __call__(self, *arrays):
    _, compiled = self._get_or_trace(arrays)
    compiled(*arrays)

  def lower(self, *arrays) -> str:
    """Return lowered NumPy source. Pass arrays for shape-aware (fused) lowering."""
    if arrays:
      ir, _ = self._get_or_trace(arrays)
      return lower_fused_numpy(ir)
    return lower_to_numpy(trace_kernel(self._fn))

  def show_ir(self, *arrays) -> str:
    """Return pretty-printed IR. Pass arrays for shape-aware IR."""
    if arrays:
      ir, _ = self._get_or_trace(arrays)
    else:
      ir = trace_kernel(self._fn)
    return pretty_print(ir)


def kernel(fn):
  """Decorator: marks a function as a mini_pallas kernel."""
  return KernelFunction(fn)
