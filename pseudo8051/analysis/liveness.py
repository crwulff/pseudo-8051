"""
analysis/liveness.py — LivenessAnalysis pass.

Backward worklist dataflow generalised over all blocks in the function.
Migrated from infer_params() / _block_use_def() in the old script.

Equations:
    LIVE_IN[B]  = upward_use[B]  ∪  (LIVE_OUT[B] − defined[B])
    LIVE_OUT[B] = ∪ LIVE_IN[S]   for each successor S
"""

from collections import deque

import idc

from pseudo8051.passes      import OptimizationPass
from pseudo8051.ir.function import Function


class LivenessAnalysis(OptimizationPass):
    """
    Backward liveness analysis.
    Populates block.live_in and block.live_out for every block.
    """

    def run(self, func: Function) -> None:
        BADADDR = idc.BADADDR
        blocks_by_ea = {b.start_ea: b for b in func.blocks}
        if not blocks_by_ea:
            return

        live_in  = {bea: frozenset() for bea in blocks_by_ea}
        live_out = {bea: frozenset() for bea in blocks_by_ea}

        worklist    = deque(blocks_by_ea)
        in_worklist = set(blocks_by_ea)

        while worklist:
            bea = worklist.popleft()
            in_worklist.discard(bea)

            block = blocks_by_ea[bea]

            new_out = frozenset().union(
                *(live_in.get(s.start_ea, frozenset())
                  for s in block.successors
                  if s.start_ea < BADADDR))

            new_in = block.upward_use | (new_out - block.defined)

            if new_in != live_in[bea] or new_out != live_out[bea]:
                live_in[bea]  = new_in
                live_out[bea] = new_out
                for pred in block.predecessors:
                    if pred.start_ea in blocks_by_ea and pred.start_ea not in in_worklist:
                        worklist.append(pred.start_ea)
                        in_worklist.add(pred.start_ea)

        # Write results back
        for bea, block in blocks_by_ea.items():
            block.live_in  = live_in[bea]
            block.live_out = live_out[bea]
