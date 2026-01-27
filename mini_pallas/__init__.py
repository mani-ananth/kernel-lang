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

  @property
  def ir(self):
    if self._ir is None:
      self._ir = trace_kernel(self._fn)
    return self._ir

  def __call__(self, *arrays):
    compiled = compile_numpy(self.ir)
    compiled(*arrays)

  def lower(self) -> str:
    return lower_to_numpy(self.ir)

  def show_ir(self) -> str:
    return pretty_print(self.ir)


def kernel(fn):
  """Decorator: marks a function as a mini_pallas kernel."""
  return KernelFunction(fn)
