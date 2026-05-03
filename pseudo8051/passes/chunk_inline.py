"""passes/chunk_inline.py — ChunkInliner pass."""

import copy
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
        # Collect all branch targets across the whole function so we know which
        # chunk entry EAs are referenced by gotos (not just by the lcall itself).
        # A chunk entry that is also a branch target must keep its Label node so
        # that _structure_flat_ifelse can fold the goto into an if body.
        all_branch_targets: set = set()
        for b in func.blocks:
            for insn in b.instructions:
                for tgt in insn.branch_targets():
                    all_branch_targets.add(tgt)

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
                # If the chunk entry EA is also a branch target (not just the
                # lcall itself), prepend a Label node to the inlined HIR so that
                # _structure_flat_ifelse can match goto targets to their code.
                entry_is_branch_target = chunk_entry.start_ea in all_branch_targets

                inline_hir = []
                if entry_is_branch_target:
                    import ida_name
                    lbl_name = chunk_entry.label
                    if not lbl_name:
                        ida_lbl = (ida_name.get_ea_name(chunk_entry.start_ea, ida_name.GN_LOCAL)
                                   or ida_name.get_name(chunk_entry.start_ea))
                        lbl_name = (ida_lbl if ida_lbl
                                    else f"label_{hex(chunk_entry.start_ea).removeprefix('0x')}")
                    inline_hir.append(Label(chunk_entry.start_ea, lbl_name))

                for cb in sorted(chunk_blocks, key=lambda b: b.start_ea):
                    for node in cb.hir:
                        if isinstance(node, ReturnStmt):
                            continue
                        # Strip the Label of the chunk entry block — it was either
                        # already prepended above (branch-target case) or is unused.
                        if isinstance(node, Label) and cb is chunk_entry:
                            continue
                        inline_hir.append(copy.deepcopy(node))

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
        from pseudo8051.passes.debug_dump import dump_pass_hir
        all_nodes = [n for b in func.blocks
                     if not getattr(b, "_absorbed", False) for n in b.hir]
        dump_pass_hir("01.chunk_inline", all_nodes, func.name)


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
