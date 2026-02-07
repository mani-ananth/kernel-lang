"""Tests for benchmark utility functions."""

import sys
import numpy as np
import pytest

# Add benchmarks to path for import
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent / "benchmarks"))

from run import load_kernel, make_arrays, bench, fmt


class TestLoadKernel:
  def test_load_kernel_success(self, tmp_path):
    """load_kernel loads a kernel from file::name spec."""
    # Create a temporary kernel file
    kernel_file = tmp_path / "test_kernel.py"
    kernel_file.write_text("""
import mini_pallas

@mini_pallas.kernel
def my_kernel(x, o):
  o[...] = x[...]
""")
    kernel = load_kernel(f"{kernel_file}::my_kernel")
    assert kernel._fn.__name__ == "my_kernel"

  def test_load_kernel_bad_spec(self):
    """Missing :: separator causes SystemExit."""
    with pytest.raises(SystemExit):
      load_kernel("no_separator_here.py")

  def test_load_kernel_missing_function(self, tmp_path):
    """Non-existent function causes SystemExit."""
    kernel_file = tmp_path / "test_kernel.py"
    kernel_file.write_text("""
import mini_pallas

@mini_pallas.kernel
def existing_kernel(x, o):
  o[...] = x[...]
""")
    with pytest.raises(SystemExit):
      load_kernel(f"{kernel_file}::nonexistent")


class TestMakeArrays:
  def test_make_arrays_count(self):
    """make_arrays generates correct number of inputs."""
    import mini_pallas

    @mini_pallas.kernel
    def k(a, b, c, o):  # 3 inputs, 1 output
      o[...] = a[...] + b[...] + c[...]

    inputs, output = make_arrays(k, size=10, ndim=1, dtype=np.float32)
    assert len(inputs) == 3

  def test_make_arrays_shape(self):
    """make_arrays generates correct shapes."""
    import mini_pallas

    @mini_pallas.kernel
    def k(x, o):
      o[...] = x[...]

    inputs, output = make_arrays(k, size=64, ndim=2, dtype=np.float64)
    assert inputs[0].shape == (64, 64)
    assert output.shape == (64, 64)

  def test_make_arrays_dtype(self):
    """make_arrays uses correct dtype."""
    import mini_pallas

    @mini_pallas.kernel
    def k(x, o):
      o[...] = x[...]

    inputs, output = make_arrays(k, size=10, ndim=1, dtype=np.float32)
    assert inputs[0].dtype == np.float32
    assert output.dtype == np.float32


class TestBench:
  def test_bench_returns_times(self):
    """bench returns a list of floats with correct length."""
    counter = [0]
    def fn():
      counter[0] += 1

    times = bench(fn, rounds=5, warmup=2)
    assert len(times) == 5
    assert all(isinstance(t, float) for t in times)
    assert counter[0] == 7  # 2 warmup + 5 rounds


class TestFmt:
  def test_fmt_microseconds(self):
    """Small times are formatted as µs."""
    times = [0.000010, 0.000012, 0.000011]  # ~10-12 µs
    result = fmt(times)
    assert "us" in result

  def test_fmt_milliseconds(self):
    """Larger times are formatted as ms."""
    times = [0.005, 0.006, 0.0055]  # ~5-6 ms
    result = fmt(times)
    assert "ms" in result
