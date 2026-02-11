"""mini_pallas — a minimal Pallas-like kernel language."""

from .core import pretty_print
from .lowering import lower_to_numpy
from .runtime import compile_numpy
from .trace import trace_kernel


class KernelFunction:
  """Wraps a user-defined kernel function with trace/lower/compile/run."""

  def __init__(self, fn):
    self._fn = fn
    self._ir = None
    self._cached_shapes = None

  def _get_ir(self, arrays=None):
    """Get or create IR, optionally with shape info from arrays."""
    if arrays is not None:
      shapes = tuple(arr.shape for arr in arrays)
      dtypes = tuple(arr.dtype for arr in arrays)
      # Check if we need to retrace due to shape change
      if self._ir is None or self._cached_shapes != shapes:
        self._ir = trace_kernel(self._fn, list(shapes), list(dtypes))
        self._cached_shapes = shapes
    elif self._ir is None:
      # Trace without shape info
      self._ir = trace_kernel(self._fn)
    return self._ir

  @property
  def ir(self):
    return self._get_ir()

  def __call__(self, *arrays):
    ir = self._get_ir(arrays)
    compiled = compile_numpy(ir)
    compiled(*arrays)

  def lower(self, *arrays) -> str:
    """Return lowered NumPy source. Pass arrays for shape-aware lowering."""
    ir = self._get_ir(arrays if arrays else None)
    return lower_to_numpy(ir)

  def show_ir(self, *arrays) -> str:
    """Return pretty-printed IR. Pass arrays for shape-aware IR."""
    ir = self._get_ir(arrays if arrays else None)
    return pretty_print(ir)


def kernel(fn):
  """Decorator: marks a function as a mini_pallas kernel."""
  return KernelFunction(fn)
