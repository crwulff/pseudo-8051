"""passes/chunk_inline.py — ChunkInliner pass."""

from typing import List, Optional

from pseudo8051.passes import OptimizationPass
from pseudo8051.ir.hir import Label, ReturnStmt
from pseudo8051.constants import dbg


class ChunkInliner(OptimizationPass):
    """
    For each lcall that targets a function tail chunk of the same function,
    inline the chunk's HIR at the call site and mark the chunk blocks absorbed.
    Runs after initial_hir() but before structural passes.

    IDA does not always split blocks at intra-function chunk calls, so the
    LCALL may appear in the middle of a block.  We therefore scan all
    instructions (not just the last) and insert the inlined HIR at the
    EA-correct position.
    """

    def run(self, func) -> None:
        for block in list(func.blocks):
            if not block.instructions:
                continue

            # Collect all chunk calls in this block, in order
            chunk_calls = []  # list of (insn_ea, chunk_entry_block)
            for insn in block.instructions:
                if not insn.is_call():
                    continue
                target_ea = _chunk_target(insn, func.ea)
                if target_ea is None:
                    continue
                chunk_entry = func._block_map.get(target_ea)
                if chunk_entry is None:
                    dbg("func", f"  ChunkInliner: chunk {hex(target_ea)} not in _block_map, skipping")
                    continue
                chunk_calls.append((insn.ea, chunk_entry))

            if not chunk_calls:
                continue

            # Process in reverse order so that earlier insertions don't shift later indices
            chunk_calls_rev = list(reversed(chunk_calls))
            for call_ea, chunk_entry in chunk_calls_rev:
                chunk_blocks = _bfs(chunk_entry)
                inline_hir = []
                for cb in sorted(chunk_blocks, key=lambda b: b.start_ea):
                    for node in cb.hir:
                        if not isinstance(node, (Label, ReturnStmt)):
                            inline_hir.append(node)

                # Find insertion point: first HIR node whose EA is strictly
                # greater than the call instruction's EA.  Since the chunk call
                # lifted to [] there is no node for it in block.hir.
                insert_pos = len(block.hir)
                for i, node in enumerate(block.hir):
                    if node.ea > call_ea:
                        insert_pos = i
                        break

                block.hir[insert_pos:insert_pos] = inline_hir
                dbg("func", f"  ChunkInliner: inlined {len(inline_hir)} nodes from "
                            f"{hex(chunk_entry.start_ea)} into block {hex(block.start_ea)} "
                            f"at pos {insert_pos} (call ea={hex(call_ea)})")
                for cb in chunk_blocks:
                    cb._absorbed = True


def _chunk_target(insn, func_ea: int) -> Optional[int]:
    """Return chunk entry EA if insn is a call to a tail chunk of func_ea, else None."""
    targets = insn.branch_targets()
    if not targets:
        return None
    target_ea = targets[0]
    import ida_funcs
    target_fn = ida_funcs.get_func(target_ea)
    if (target_fn is None
            or target_fn.start_ea != func_ea   # must be owned by this function
            or target_ea == func_ea):           # chunk entry = function entry, not a chunk
        return None
    return target_ea


def _bfs(entry) -> list:
    """BFS from entry through successors within the same function."""
    visited, seen, queue = [], set(), [entry]
    while queue:
        b = queue.pop(0)
        if b.start_ea in seen:
            continue
        seen.add(b.start_ea)
        visited.append(b)
        queue.extend(b.successors)
    return visited
