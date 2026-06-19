# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install in editable mode (required before running anything)
pip install -e .

# Run all tests
pytest

# Run a single test file
pytest tests/test_integration.py

# Run a single test by name
pytest tests/test_integration.py::test_vector_add

# Run an example
python examples/01_vector_add.py
```

## Architecture

`mini_pallas` is a minimal Pallas-like kernel language that compiles Python functions into executable NumPy code. The pipeline is:

```
@kernel fn  →  trace_kernel  →  KernelIR  →  find_fusion_groups  →  lower_fused_numpy  →  exec()  →  callable
```

**`core.py`** — IR definitions. `KernelIR` holds a list of `IROp`s in SSA form. Each `IRValue` has a unique integer ID, name, shape, and dtype. `OpType` enumerates all operations (LOAD, STORE, CONST, ADD, SUB, MUL, TRUEDIV, NEG, MATMUL).

**`trace.py`** — Tracing. `trace_kernel(fn)` calls the user's function with `TracerRef` proxies (one per parameter). Indexing a `TracerRef` with `[...]` emits LOAD/STORE ops; arithmetic on `TracerValue`s emits the corresponding binary ops. Shape/dtype propagation happens here (broadcasting and matmul shape inference).

**`passes.py`** — Fusion analysis. `find_fusion_groups(ir)` walks backward from each STORE op, absorbing single-use element-wise ops (ADD/SUB/MUL/TRUEDIV/NEG), LOADs, and CONSTs into a `FusionGroup`. MATMUL and multi-use values remain as external inputs. Groups require a known non-scalar `output_shape`.

**`lowering.py`** — Code generation. `lower_to_numpy(ir)` produces array-level NumPy code. `lower_fused_numpy(ir)` replaces each `FusionGroup` with explicit nested `for` loops using broadcast-aware index expressions.

**`runtime.py`** — Execution. `compile_fused_numpy(ir)` calls `lower_fused_numpy`, `exec`s the source string, and caches the resulting callable keyed by `id(ir)`.

**`__init__.py`** — Public API. The `@kernel` decorator wraps a function in `KernelFunction`, which lazily traces and retraces when input shapes change. `KernelFunction.__call__(*arrays)` compiles with fusion and runs in-place.

## Key design conventions

- Kernel functions take only `ref` parameters (no return value); the last parameter is conventionally the output ref.
- `ref[...]` (Ellipsis indexing only) is the only supported indexing — no slicing or integer indices.
- Kernels are traced once per unique set of input shapes; shape changes trigger retrace.
- `lower()` without arrays produces unfused (array-level) code; `lower(*arrays)` produces fused (loop-level) code.
