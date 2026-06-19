"""mini_pallas — a minimal Pallas-like kernel language."""

from .core import pretty_print
from .lowering import lower_to_numpy
from .mlx_lowering import lower_to_mlx
from .runtime import compile_mlx, compile_numpy
from .trace import trace_kernel


class KernelFunction:
  """Wraps a user-defined kernel function with trace/lower/compile/run.

  Guard-based retracing: each unique (shapes, dtypes) signature gets its own
  cached (KernelIR, compiled_fn) entry, mirroring JAX/Dynamo's approach.

  backend: "numpy" (default, vectorized C via out= ufuncs) or "mlx" (Apple GPU).
  """

  def __init__(self, fn, backend: str = "numpy"):
    self._fn = fn
    self._backend = backend
    self._guard_cache: dict[tuple, tuple] = {}

  def _guard_key(self, arrays) -> tuple:
    shapes = tuple(arr.shape for arr in arrays)
    dtypes = tuple(arr.dtype for arr in arrays)
    return (shapes, dtypes)

  def _compile(self, ir):
    if self._backend == "mlx":
      return compile_mlx(ir)
    return compile_numpy(ir)

  def _get_or_trace(self, arrays) -> tuple:
    key = self._guard_key(arrays)
    if key not in self._guard_cache:
      shapes = [arr.shape for arr in arrays]
      dtypes = [arr.dtype for arr in arrays]
      ir = trace_kernel(self._fn, shapes, dtypes)
      self._guard_cache[key] = (ir, self._compile(ir))
    return self._guard_cache[key]

  @property
  def ir(self):
    if self._guard_cache:
      return next(iter(self._guard_cache.values()))[0]
    return trace_kernel(self._fn)

  def __call__(self, *arrays):
    _, compiled = self._get_or_trace(arrays)
    compiled(*arrays)

  def lower(self, *arrays) -> str:
    """Return lowered source. Pass arrays for shape-aware tracing."""
    if arrays:
      ir, _ = self._get_or_trace(arrays)
      if self._backend == "mlx":
        return lower_to_mlx(ir)
      return lower_to_numpy(ir)
    ir = trace_kernel(self._fn)
    if self._backend == "mlx":
      return lower_to_mlx(ir)
    return lower_to_numpy(ir)

  def show_ir(self, *arrays) -> str:
    if arrays:
      ir, _ = self._get_or_trace(arrays)
    else:
      ir = trace_kernel(self._fn)
    return pretty_print(ir)


def kernel(fn=None, *, backend: str = "numpy"):
  """Decorator: marks a function as a mini_pallas kernel.

  Usage:
    @kernel
    def fn(...): ...              # NumPy backend (default)

    @kernel(backend="mlx")
    def fn(...): ...              # MLX backend (Apple GPU via Metal)
  """
  if fn is not None:
    return KernelFunction(fn, backend=backend)
  def decorator(f):
    return KernelFunction(f, backend=backend)
  return decorator
