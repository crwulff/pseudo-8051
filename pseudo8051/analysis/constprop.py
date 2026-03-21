"""
analysis/constprop.py — ConstantPropagation pass.

Drives the fixed-point constant-propagation iteration over the block graph.
CPState and propagate_insn() live in ir/cpstate.py (they are IR data, not
an analysis algorithm).
"""

from collections import deque
from typing import Dict

import ida_ua
import ida_bytes
import idc

from pseudo8051.ir.cpstate import CPState, propagate_insn  # noqa: F401 (re-exported)
from pseudo8051.ir.function import Function
from pseudo8051.passes      import OptimizationPass


# ── Analysis pass ─────────────────────────────────────────────────────────────

class ConstantPropagation(OptimizationPass):
    """
    Forward constant-propagation pass.
    Populates block.cp_entry (CPState at the START of each block).
    """

    def run(self, func: Function) -> None:
        BADADDR = idc.BADADDR
        blocks_by_ea = {b.start_ea: b for b in func.blocks}

        entry_states: Dict[int, CPState] = {func.ea: CPState()}
        exit_states:  Dict[int, CPState] = {}

        worklist    = deque([func.ea])
        in_worklist = {func.ea}

        while worklist:
            bea = worklist.popleft()
            in_worklist.discard(bea)

            block = blocks_by_ea.get(bea)
            if block is None:
                continue

            # Entry state = meet of predecessor exit states
            if bea == func.ea:
                new_entry = CPState()
            else:
                pred_exits = [exit_states[p.start_ea]
                              for p in block.predecessors
                              if p.start_ea in exit_states]
                new_entry = CPState.meet(pred_exits) if pred_exits else CPState()

            if entry_states.get(bea) == new_entry:
                continue

            entry_states[bea] = new_entry

            # Thread state through the block's instructions
            cur = new_entry.copy()
            ea  = block.start_ea
            while ea < block.end_ea and ea < BADADDR:
                insn = ida_ua.insn_t()
                size = ida_ua.decode_insn(insn, ea)
                if size > 0:
                    propagate_insn(insn, cur)
                    ea += size
                else:
                    nxt = ida_bytes.next_head(ea, block.end_ea)
                    if nxt <= ea or nxt >= BADADDR:
                        break
                    ea = nxt

            old_exit = exit_states.get(bea)
            exit_states[bea] = cur
            if old_exit != cur:
                for succ in block.successors:
                    if succ.start_ea < BADADDR and succ.start_ea not in in_worklist:
                        worklist.append(succ.start_ea)
                        in_worklist.add(succ.start_ea)

        # Write results back to blocks
        for bea, block in blocks_by_ea.items():
            block.cp_entry = entry_states.get(bea, CPState())
