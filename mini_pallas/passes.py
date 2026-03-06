"""Fusion analysis: identify element-wise op chains that can be fused into loops."""

from dataclasses import dataclass, field

from .core import KernelIR, OpType


ELEMENTWISE_OPS = {OpType.ADD, OpType.SUB, OpType.MUL, OpType.TRUEDIV, OpType.NEG}


def _use_counts(ir: KernelIR) -> dict[int, int]:
  """Map IRValue.id -> number of times consumed as an operand."""
  counts: dict[int, int] = {}
  for op in ir.ops:
    for operand in op.operands:
      counts[operand.id] = counts.get(operand.id, 0) + 1
  return counts


def _producer_map(ir: KernelIR) -> dict[int, int]:
  """Map IRValue.id -> index in ir.ops that produced it."""
  producers: dict[int, int] = {}
  for i, op in enumerate(ir.ops):
    if op.result is not None:
      producers[op.result.id] = i
  return producers


@dataclass
class FusionGroup:
  """A group of ops that can be fused into a single nested loop."""
  fused_op_indices: list[int] = field(default_factory=list)
  store_index: int = -1
  load_indices: list[int] = field(default_factory=list)
  const_indices: list[int] = field(default_factory=list)
  output_shape: tuple[int, ...] = ()


def find_fusion_groups(ir: KernelIR) -> list[FusionGroup]:
  """Walk backward from each STORE, absorbing single-use element-wise ops, LOADs, and CONSTs.

  Rules:
  - An element-wise op is absorbed if its result has exactly one consumer.
  - A LOAD/CONST is absorbed if single-use and consumed only within the group.
  - MATMUL results and multi-use values are "external inputs" — left materialized.
  - Groups must have non-empty output_shape (needs shape-aware tracing).
  - Groups must contain at least 1 element-wise op.
  """
  uses = _use_counts(ir)
  producers = _producer_map(ir)

  claimed: set[int] = set()  # op indices already in a group
  groups: list[FusionGroup] = []

  for store_idx, op in enumerate(ir.ops):
    if op.op_type != OpType.STORE:
      continue
    if store_idx in claimed:
      continue

    # The value being stored
    stored_val = op.operands[0]

    # Walk backward from the stored value, collecting fusable ops
    fused_indices: list[int] = []
    load_indices: list[int] = []
    const_indices: list[int] = []
    group_value_ids: set[int] = set()  # value IDs produced within this group

    # BFS/DFS worklist: value IDs to explore
    worklist = [stored_val.id]

    while worklist:
      val_id = worklist.pop()
      if val_id not in producers:
        continue
      prod_idx = producers[val_id]
      if prod_idx in claimed:
        continue
      prod_op = ir.ops[prod_idx]

      if prod_op.op_type in ELEMENTWISE_OPS:
        # Absorb if single-use
        if uses.get(val_id, 0) == 1:
          fused_indices.append(prod_idx)
          group_value_ids.add(val_id)
          # Recurse into operands
          for operand in prod_op.operands:
            worklist.append(operand.id)
        # Multi-use: leave as external input

      elif prod_op.op_type == OpType.LOAD:
        if uses.get(val_id, 0) == 1:
          load_indices.append(prod_idx)
          group_value_ids.add(val_id)

      elif prod_op.op_type == OpType.CONST:
        if uses.get(val_id, 0) == 1:
          const_indices.append(prod_idx)
          group_value_ids.add(val_id)

      # MATMUL or anything else: leave as external input

    if not fused_indices:
      continue

    # Determine output shape from the stored value
    output_shape = stored_val.shape
    if output_shape is None or len(output_shape) == 0:
      # Try to infer from the store's ref
      # Can't fuse without shape info
      continue

    # Sort indices into topological (original) order
    fused_indices.sort()
    load_indices.sort()
    const_indices.sort()

    group = FusionGroup(
      fused_op_indices=fused_indices,
      store_index=store_idx,
      load_indices=load_indices,
      const_indices=const_indices,
      output_shape=output_shape,
    )
    groups.append(group)

    # Mark all indices as claimed
    claimed.add(store_idx)
    claimed.update(fused_indices)
    claimed.update(load_indices)
    claimed.update(const_indices)

  return groups
