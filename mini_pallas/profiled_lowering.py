"""Profiling-instrumented lowering variants.

Generates the same code as lowering.py / mlx_lowering.py but wraps each
operation with _p.complete() timing calls. The profiler is passed via the
exec() namespace as `_p`, so generated functions call it directly.

These are for analysis only — not used in the normal execution path.
"""

from typing import Callable

import numpy as np

from .core import KernelIR, OpType
from .lowering import _NUMPY_UFUNC, _format_const

# ──────────────────────────────────────────────────────────────────────────────
# Timing injection helper
# ──────────────────────────────────────────────────────────────────────────────

_T = "_t"  # local variable reused for each timestamp capture


def _timed(stmt: str, label: str, cat: str, indent: str = "  ") -> str:
    """Wrap a statement with _p.complete() timing on a single line."""
    return (
        f"{indent}{_T} = _p._ts(); "
        f"{stmt}; "
        f"_p.complete({label!r}, {_T}, _p._ts()-{_T}, cat={cat!r})"
    )


# ──────────────────────────────────────────────────────────────────────────────
# NumPy profiled lowering
# ──────────────────────────────────────────────────────────────────────────────

def lower_to_numpy_profiled(ir: KernelIR) -> str:
    """Like lower_to_numpy but injects _p.complete() around each ufunc call.

    The generated function reads `_p` from its exec() namespace.
    """
    use_count: dict[int, int] = {}
    for op in ir.ops:
        for v in op.operands:
            use_count[v.id] = use_count.get(v.id, 0) + 1

    alias: dict[int, str] = {
        op.result.id: op.ref_name
        for op in ir.ops if op.op_type == OpType.LOAD
    }

    store_op = next(op for op in ir.ops if op.op_type == OpType.STORE)
    store_ref = store_op.ref_name
    store_val_id = store_op.operands[0].id
    store_val_is_alias = store_val_id in alias

    def resolve(v) -> str:
        return alias.get(v.id, v.name)

    def out_for(result_id: int) -> str | None:
        if result_id == store_val_id and not store_val_is_alias:
            return store_ref
        if use_count.get(result_id, 0) == 1:
            return "_buf"
        return None

    needs_buf = any(
        op.result
        and op.result.id != store_val_id
        and op.result.id not in alias
        and use_count.get(op.result.id, 0) == 1
        and op.op_type not in (OpType.LOAD, OpType.CONST)
        for op in ir.ops
    )

    sig = ", ".join(ir.ref_params)
    lines = [f"def {ir.name}({sig}):"]

    if needs_buf:
        lines.append(_timed(f"_buf = np.empty_like({store_ref})", "empty_like", "alloc"))

    for op in ir.ops:
        if op.op_type in (OpType.LOAD, OpType.STORE):
            continue
        if op.op_type == OpType.CONST:
            lines.append(f"  {op.result.name} = {_format_const(op.const_value)}")
            continue

        fn = _NUMPY_UFUNC[op.op_type]
        args = ", ".join(resolve(v) for v in op.operands)
        out = out_for(op.result.id)

        if out:
            is_final = op.result.id == store_val_id
            call = f"{fn}({args}, out={out})"
            stmt = call if is_final else f"{op.result.name} = {call}"
            lines.append(_timed(stmt, fn, "op"))
        else:
            stmt = f"{op.result.name} = {fn}({args})"
            lines.append(_timed(stmt, fn, "op"))

    if store_val_is_alias:
        stmt = f"{store_ref}[...] = {alias[store_val_id]}"
        lines.append(_timed(stmt, "copy", "op"))

    return "\n".join(lines)


def compile_numpy_profiled(ir: KernelIR, profiler) -> Callable:
    """Compile a profiled NumPy function. `profiler` is bound as `_p`."""
    source = lower_to_numpy_profiled(ir)
    namespace: dict = {"np": np, "_p": profiler}
    exec(source, namespace)
    return namespace[ir.name]


# ──────────────────────────────────────────────────────────────────────────────
# MLX profiled lowering
# ──────────────────────────────────────────────────────────────────────────────

# Category labels that make the trace informative at a glance
_MLX_OP_LABELS = {
    OpType.ADD: "add (lazy)",
    OpType.SUB: "sub (lazy)",
    OpType.MUL: "mul (lazy)",
    OpType.TRUEDIV: "div (lazy)",
    OpType.MATMUL: "matmul (lazy)",
    OpType.NEG: "neg (lazy)",
}

_MLX_BINOP = {
    OpType.ADD: "+",
    OpType.SUB: "-",
    OpType.MUL: "*",
    OpType.TRUEDIV: "/",
    OpType.MATMUL: "@",
}


def lower_to_mlx_profiled(ir: KernelIR) -> str:
    """Like lower_to_mlx but injects _p.complete() with semantic categories:

      h2d       — mx.array() host-to-device transfers
      op_lazy   — lazy graph construction (no GPU work yet)
      sync      — mx.eval() where GPU computation actually runs
      d2h       — np.array() device-to-host writeback
    """
    sig = ", ".join(ir.ref_params)
    lines = [f"def {ir.name}({sig}):"]

    for op in ir.ops:
        if op.op_type == OpType.LOAD:
            ref = op.ref_name
            stmt = f"{op.result} = mx.array({ref})"
            lines.append(_timed(stmt, f"mx.array({ref})", "h2d"))

        elif op.op_type == OpType.STORE:
            stored = op.operands[0]
            lines.append(_timed(f"mx.eval({stored})", "mx.eval", "sync"))
            lines.append(_timed(f"{op.ref_name}[...] = np.array({stored})", "np.array (d2h)", "d2h"))

        elif op.op_type == OpType.CONST:
            # Scalar constants are trivial; array constants are a host-side alloc
            val = _format_const(op.const_value)
            val_mx = val.replace("np.array", "mx.array")
            lines.append(f"  {op.result.name} = {val_mx}")

        elif op.op_type in _MLX_BINOP:
            lhs, rhs = op.operands
            sym = _MLX_BINOP[op.op_type]
            label = _MLX_OP_LABELS[op.op_type]
            stmt = f"{op.result} = {lhs} {sym} {rhs}"
            lines.append(_timed(stmt, label, "op_lazy"))

        elif op.op_type == OpType.NEG:
            label = _MLX_OP_LABELS[op.op_type]
            stmt = f"{op.result} = -{op.operands[0]}"
            lines.append(_timed(stmt, label, "op_lazy"))

        else:
            raise ValueError(f"Unsupported op: {op.op_type}")

    return "\n".join(lines)


def compile_mlx_profiled(ir: KernelIR, profiler) -> Callable:
    """Compile a profiled MLX function. `profiler` is bound as `_p`."""
    import mlx.core as mx

    source = lower_to_mlx_profiled(ir)
    namespace: dict = {"mx": mx, "np": np, "_p": profiler}
    exec(source, namespace)
    return namespace[ir.name]
