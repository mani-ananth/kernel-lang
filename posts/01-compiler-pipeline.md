---
title: "Building a Kernel Compiler in 500 Lines of Python"
subtitle: "How a Python function becomes machine-executable code — the tracing, IR, and lowering tricks that JAX, Triton, and torch.compile all share."
date: 2026-06-20
tags: [compilers, python, jax, pallas]
---

When you write `o = a * b + c` in NumPy, something straightforward happens: NumPy allocates a temporary array for `a * b`, then another for the final result. Two passes over memory, two allocations, one answer.

When you write the same line in JAX or `torch.compile`, something far less straightforward happens. The framework intercepts your function before it runs, builds a representation of what you *intended* to compute, and then generates code that does it differently — often in one fused pass with no temporaries in main memory.

This post is about how that interception works. To make the ideas concrete, I built [`picokernel`](https://github.com/mani-ananth/picokernel) — a ~500-line kernel compiler in pure Python. It's a teaching toy, not a production system, but it implements the same core ideas as JAX Pallas, Triton, and `torch.compile`: tracing with proxy objects, an SSA intermediate representation, and a lowering pass that emits the actual executable code.

Here's the whole pipeline:

![Diagram: the picokernel pipeline from decorated Python function through trace_kernel, KernelIR, lower_to_numpy, exec, to a cached callable. Arrows show data flow with labels.](./images/01-pipeline-overview.png)

I'll walk through each arrow.

---

## The kernel contract

Before any of the machinery makes sense, you need to know what a "kernel" looks like in this system:

```python
import picokernel

@picokernel.kernel
def fma(a_ref, b_ref, c_ref, o_ref):
    o_ref[...] = a_ref[...] * b_ref[...] + c_ref[...]
```

Three rules, all borrowed from real kernel languages:

1. **Every parameter is a reference (`ref`).** There's no return value. The kernel writes its result into one of the output refs — by convention, the last one.
2. **The only legal indexing is `ref[...]`.** No slicing, no integer indices. The ellipsis means "load the whole thing" on the right side, "store into the whole thing" on the left.
3. **The function body is straight-line numeric code.** No control flow over data values, no Python objects flowing through arithmetic.

These rules feel restrictive — and they are — but every restriction buys something specific from the compiler. Refs make data flow explicit. Ellipsis-only indexing eliminates an entire class of stride and aliasing analyses. Straight-line code means tracing can capture the entire computation in a single pass. Together, they're what makes a 500-line compiler possible at all — every relaxation costs lines of compiler.

None of them are permanent, though. They're a floor, not a ceiling, and each one maps to a specific extension at a known cost. Replacing `ref[...]` with `ref[i:j]` is roughly a day of work. Tiled access like `ref[i:i+BM, :]` inside a `for` loop needs a `FOR` op in the IR and is where C or Metal codegen takes over — the topic of the next post in this series. The straight-line rule has a useful split too: static loops (`for i in range(4):` with a literal trip count) and `if`s on Python-side values already work, because Python unrolls them before the tracer sees them. The line tracing can't cross is *data-dependent* control flow (`if a_ref[0] > 0:`) — the same problem JAX solves with `lax.cond` and Pallas with `pl.when`, by deferring the decision into the IR as a new op type rather than resolving it at trace time.

JAX Pallas, Triton, and CUDA all impose similar contracts. The exact shape varies, but the spirit is the same: *give up some Python flexibility and the compiler can do dramatic things with what remains*.

---

## Step 1: Tracing with proxy objects

The first thing the compiler needs to do is *figure out what the function computes*. There are two ways to do this:

- **Parse the source.** Read the Python AST, interpret it symbolically. This is what compilers like Numba do.
- **Run the function.** But run it with fake arrays — proxy objects that record every operation performed on them.

`picokernel` does the second. It's called *tracing*, and it's how JAX, PyTorch's `torch.compile`, and TensorFlow's autograph all work. The trick is to never actually do the arithmetic; instead, every operation appends to a log.

Here's the proxy in its simplest form:

```python
class TracerRef:
    """A fake array reference. Indexing it emits a LOAD or STORE."""
    def __init__(self, name, shape, dtype):
        self.name = name
        self.shape = shape
        self.dtype = dtype

    def __getitem__(self, key):
        # key is always Ellipsis (...)
        return emit_load(self)

    def __setitem__(self, key, value):
        emit_store(self, value)


class TracerValue:
    """A fake array value. Arithmetic on it emits ops."""
    def __init__(self, value_id, shape, dtype):
        self.value_id = value_id
        self.shape = shape
        self.dtype = dtype

    def __mul__(self, other):
        return emit_binop("MUL", self, other)

    def __add__(self, other):
        return emit_binop("ADD", self, other)
```

To trace a kernel, you build a fresh `TracerRef` for each parameter and call the user's function:

```python
def trace_kernel(fn, shapes, dtypes):
    refs = [TracerRef(name, shape, dtype)
            for name, shape, dtype in zip(fn.__code__.co_varnames, shapes, dtypes)]
    fn(*refs)
    return current_ir()
```

When the user writes `o_ref[...] = a_ref[...] * b_ref[...] + c_ref[...]`, Python evaluates that line by:

1. Calling `a_ref.__getitem__(...)` — emits `v0 = LOAD a_ref`, returns a `TracerValue`.
2. Calling `b_ref.__getitem__(...)` — emits `v1 = LOAD b_ref`.
3. Calling `v0.__mul__(v1)` — emits `v2 = MUL(v0, v1)`.
4. Calling `c_ref.__getitem__(...)` — emits `v3 = LOAD c_ref`.
5. Calling `v2.__add__(v3)` — emits `v4 = ADD(v2, v3)`.
6. Calling `o_ref.__setitem__(..., v4)` — emits `STORE o_ref ← v4`.

The user's function never knows it was being watched. Python's operator overloading does all the work.

![Diagram: the line `o_ref[...] = a_ref[...] * b_ref[...] + c_ref[...]` broken down into the six dunder method calls (__getitem__, __mul__, __getitem__, __add__, __setitem__) and the IR ops they emit. Shows how Python's operator overloading drives trace construction.](./images/01-trace-flow.png)

**What you give up:** any control flow that depends on actual data values is invisible to tracing. If the user writes `if a_ref[0] > 0:`, the tracer can't see what's inside the `if` branch unless it's actually taken. This is why JAX has special `lax.cond` and `lax.scan` primitives: they let you express control flow in a way that *can* be traced.

**What you gain:** the trace runs the user's actual Python. Helper functions, list comprehensions, even `numpy` calls on non-tracer values — anything that doesn't touch the tracer flows through normally. The kernel author writes Python; the compiler sees a clean trace.

---

## Step 2: The IR

The trace produces a `KernelIR` — a list of operations in **Static Single Assignment** (SSA) form. SSA is a piece of compiler vocabulary that sounds intimidating but means something simple: every value gets a unique name and is assigned exactly once.

Here's the IR for our `fma` kernel:

```
kernel fma(a, b, c, o):
  v0<4:float32> = LOAD [a]()
  v1<4:float32> = LOAD [b]()
  v2<4:float32> = MUL(v0, v1)
  v3<4:float32> = LOAD [c]()
  v4<4:float32> = ADD(v2, v3)
  STORE [o](v4)
```

Each line is one operation. Each `vN` is a value, born once, never reassigned. The angle brackets carry the inferred shape and dtype.

Why bother with this representation instead of working from the Python source directly? Because SSA makes analysis trivial:

- **Want to know how many times `v2` is used?** Walk the ops once, count appearances. (Answer: once, in the `ADD`.)
- **Want to know which loads can be eliminated?** Any `LOAD` whose value is never read.
- **Want to fuse two operations?** Check that the producer's value has exactly one consumer.

In NumPy or PyTorch source, the same question — "is this intermediate used elsewhere?" — would require parsing scope, tracking aliases, handling reassignments. In SSA, it's a single pass.

This is the same form LLVM uses internally, the same form JAX's `jaxpr` uses, the same form GCC's GIMPLE uses. It's compiler infrastructure 101 for a reason.

---

## Step 3: Lowering — and a 800x speedup hiding in plain sight

Now the interesting part. We have the IR. We need to turn it into code that actually runs.

The naive approach is to walk the ops and emit one line per op:

```python
# Naive lowering of fma
def fma(a, b, c, o):
    v0 = a       # LOAD a
    v1 = b       # LOAD b
    v2 = v0 * v1 # MUL — allocates a temporary
    v3 = c       # LOAD c
    v4 = v2 + v3 # ADD — allocates another temporary
    o[...] = v4  # STORE
```

This is exactly what writing the line directly in NumPy does. Two allocations, two passes over memory. It works, but it leaves performance on the table.

The clever move: NumPy's universal functions (`np.add`, `np.multiply`, etc.) accept an `out=` keyword argument that writes the result directly into a pre-existing buffer. No allocation, no intermediate. If you wire those `out=` arguments correctly, you can compute `a * b + c` with **zero intermediate arrays** — only the user's pre-allocated output, plus one scratch buffer reused across all intermediate ops.

Here's what `picokernel` emits:

```python
def fma(a, b, c, o):
    _buf = np.empty_like(o)             # one scratch buffer, allocated once
    v2 = np.multiply(a, b, out=_buf)    # writes into _buf, no new allocation
    np.add(v2, c, out=o)                # writes directly into the output ref
```

Two operations, one buffer, zero waste.

![Side-by-side comparison: left panel shows naive lowering with two intermediate allocations highlighted; right panel shows out= lowering with arrows showing buffer reuse from _buf, and the final operation writing directly into the output ref o.](./images/01-out-lowering.png)

The lowering algorithm to produce this is short. In plain English:

1. **LOADs become aliases.** `v0 = LOAD a` doesn't generate code — instead, every subsequent use of `v0` is rewritten to use `a` directly. NumPy operations don't care that the operand "came from a load"; they just need the array.
2. **Count uses of every intermediate value.** Anything used exactly once is a candidate for the shared scratch buffer.
3. **For each compute op, pick the output destination:**
   - If this op produces the value stored at the end, use the output ref directly.
   - Otherwise, if this op's result is used exactly once, use `_buf`.
   - Otherwise (multiple uses), let NumPy allocate fresh.
4. **The final op writes directly into the output ref** — no extra store needed.

That last step is the punchline. The "STORE" op in the IR doesn't compile to a copy. It vanishes into the `out=` of whatever produced the value being stored.

The kicker: an earlier version of `picokernel` lowered the IR to a Python `for` loop over array elements — conceptually pure, no intermediate arrays. It was **100 to 800 times slower** than the `out=` version. Python's per-iteration interpreter overhead is catastrophic for numeric work. The compiler's job isn't just to eliminate allocations; it's to stay in vectorized C the whole way down.

This is the same insight that drives kernel fusion in XLA, TVM, and Triton: avoid materializing intermediates, but also avoid leaving the fast lane to do it.

---

## Step 4: Guard-based retracing

Tracing happens once per shape. The first time you call:

```python
fma(np.ones(4), np.ones(4), np.ones(4), np.zeros(4))
```

`picokernel` traces, generates code, compiles it with `exec()`, and caches the result. The cache key is `(shapes, dtypes)`:

```python
key = (
    tuple(arr.shape for arr in arrays),    # ((4,), (4,), (4,), (4,))
    tuple(arr.dtype for arr in arrays),    # (float32, float32, float32, float32)
)
```

Call it again with the same shapes — instant cache hit. Call it with shape `(8,)` — a miss, and a fresh trace. Shape `(4, 4)` — another miss, another trace.

This is the *guard* model. JAX calls them "tracing dispatch keys." `torch.compile` calls them "guards." They serve the same purpose: the compiled code is specialized to specific shapes, and any deviation forces a recompile.

This is also why these systems can feel mysteriously slow on the first call and instant on subsequent calls — and why benchmarks need a "warmup" phase. The first call pays the trace + compile cost; later calls only pay the execution cost.

---

## What this teaches

`picokernel` is a toy. It supports a handful of element-wise ops, one backend (NumPy) plus an Apple GPU experiment, no autodiff, no fusion across kernels, no memory hierarchy modeling. JAX Pallas, Triton, and `torch.compile` are vastly more sophisticated.

But the architecture is the same. The same five-step dance happens every time you decorate a function with `@jax.jit` or `@torch.compile`:

1. A proxy-based tracer captures the computation as data flow, not Python code.
2. The trace is normalized into SSA-form IR.
3. A lowering pass emits target-specific code, exploiting the IR's structure to skip work the naive version would do.
4. The result is cached, keyed by shape and dtype.
5. Subsequent calls are nearly free.

If you've ever wondered why `jax.jit` makes things faster, or why `torch.compile` is sometimes brittle around dynamic shapes, or what the difference is between "tracing" and "scripting" — this is the substrate underneath all of it.

The next post in this series digs into the GPU backend, where I tried to beat NumPy with Apple's Metal-backed MLX. Spoiler: NumPy won every round. The reasons are more interesting than the result.

---

*Code: [github.com/mani-ananth/picokernel](https://github.com/mani-ananth/picokernel)*

*Next post: [NumPy vs MLX on Apple Silicon: where GPU acceleration actually helps](./02-numpy-vs-mlx.md)*
