"""passes/simple_inline.py — SimpleExternalInliner pass.

Inline very simple external callees at their call sites.  A callee is
"simple" if it consists of a single basic block with no branches or nested
calls, at most MAX_INSTRUCTIONS assembly instructions, and ends with a plain
RET.  The callee's raw HIR (minus the ReturnStmt) is substituted for the
call node so the TypeAwareSimplifier can work on the combined code in context.

Runs immediately after ChunkInliner, before structural passes.
"""

import copy
from typing import Dict, List, Optional

from pseudo8051.passes import OptimizationPass
from pseudo8051.ir.hir import ReturnStmt, Assign, ExprStmt
from pseudo8051.ir.cpstate import CPState, propagate_insn
from pseudo8051.ir.instruction import Instruction
from pseudo8051.constants import dbg

MAX_INSTRUCTIONS = 10


class SimpleExternalInliner(OptimizationPass):
    """
    For each LCALL/ACALL to a simple external function, replace the call
    node with the callee's raw HIR body (ReturnStmt stripped).
    Runs after ChunkInliner, before structural passes.
    """

    def run(self, func) -> None:
        # Cache lifted callee bodies (None = not simple / already checked)
        _cache: Dict[int, Optional[List]] = {}

        for block in list(func.blocks):
            if not block.instructions:
                continue

            # Collect external call sites in this block, in order
            ext_calls = []
            for insn in block.instructions:
                if not insn.is_call():
                    continue
                targets = insn.branch_targets()
                if not targets:
                    continue
                target_ea = targets[0]
                if _is_chunk(target_ea, func.ea):
                    continue   # already handled by ChunkInliner
                ext_calls.append((insn.ea, target_ea))

            if not ext_calls:
                continue

            # Process in reverse order so earlier replacements don't shift later indices
            for call_ea, callee_ea in reversed(ext_calls):
                if callee_ea not in _cache:
                    _cache[callee_ea] = _lift_simple_callee(callee_ea)
                inline_nodes = _cache[callee_ea]
                if inline_nodes is None:
                    continue

                # Find the HIR node produced by this call (matched by EA and Call content)
                for idx, node in enumerate(block.hir):
                    if node.ea == call_ea and _contains_call(node):
                        block.hir[idx : idx + 1] = [copy.deepcopy(n) for n in inline_nodes]
                        dbg("func", f"  SimpleInliner: inlined {len(inline_nodes)} nodes "
                                    f"from {hex(callee_ea)} at call {hex(call_ea)}")
                        break

        from pseudo8051.passes.debug_dump import dump_pass_hir
        all_nodes = [n for b in func.blocks
                     if not getattr(b, "_absorbed", False) for n in b.hir]
        dump_pass_hir("01b.simple_inline", all_nodes, func.name)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_chunk(target_ea: int, func_ea: int) -> bool:
    """True if target_ea is an intra-function tail chunk (not an external callee)."""
    import ida_funcs
    target_fn = ida_funcs.get_func(target_ea)
    return (target_fn is not None
            and target_fn.start_ea == func_ea
            and target_ea != func_ea)


def _contains_call(node) -> bool:
    """True if node is an Assign or ExprStmt whose expression is a Call."""
    from pseudo8051.ir.expr import Call
    if isinstance(node, Assign):
        return isinstance(node.rhs, Call)
    if isinstance(node, ExprStmt):
        return isinstance(node.expr, Call)
    return False


def _lift_simple_callee(callee_ea: int) -> Optional[List]:
    """
    Lift the callee's assembly into raw HIR nodes.

    Returns the body nodes (final ReturnStmt stripped) if the callee is simple,
    or None if it exceeds the complexity threshold or cannot be inlined.
    """
    import ida_funcs
    import ida_gdl

    callee_fn = ida_funcs.get_func(callee_ea)
    if callee_fn is None or callee_fn.start_ea != callee_ea:
        return None

    blocks = list(ida_gdl.FlowChart(callee_fn))
    if len(blocks) != 1:
        return None   # has branches — not simple

    blk = blocks[0]
    state = CPState()
    nodes: List = []
    ea = blk.start_ea
    count = 0

    while ea < blk.end_ea:
        instr = Instruction(ea)
        if not instr.insn:
            return None
        if instr.is_branch() or instr.is_call():
            return None   # unexpected branch or nested call
        stmts = instr.lift(state)
        propagate_insn(instr.insn, state)
        nodes.extend(stmts)
        ea += instr.size
        count += 1
        if count > MAX_INSTRUCTIONS:
            return None

    if not nodes or not isinstance(nodes[-1], ReturnStmt):
        return None

    return nodes[:-1]   # strip ReturnStmt; caller deep-copies before insertion
