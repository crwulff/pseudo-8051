"""
passes/dead_assign.py — DeadAssignEliminator: remove HIR Assign nodes whose LHS
register is provably dead and whose RHS is side-effect-free.

Algorithm (single backward pass per BasicBlock):
  • Seed the live set from block.live_out.
  • Walk HIR backward.
  • For Assign(Reg(r), rhs) where r ∉ live and rhs is pure: mark for removal
    and skip the live-set update (as if the node doesn't exist).
  • For all kept nodes: live = (live − written_regs) ∪ name_refs().

Cascade removal is free: if node B is removed (its reads not added to live),
earlier node A that only fed B will also have its LHS dead when reached.

Runs after CopyPropagator so that assignments whose only use was a propagated
copy are caught here.

"Pure" means: no side-effect registers (no ++/--), no function call, no memory
read (XRAMRef/IRAMRef/CROMRef).  Memory writes (LHS = XRAMRef) are never removed
because the LHS is not a Reg, so written_regs is empty for those nodes.
"""

from pseudo8051.ir.function import Function
from pseudo8051.ir.hir      import HIRNode, Assign
from pseudo8051.ir.expr     import Expr, Regs, Call, XRAMRef, IRAMRef, CROMRef
from pseudo8051.passes      import OptimizationPass


def _is_pure(expr: Expr) -> bool:
    """True if evaluating expr has no observable side effects and reads no memory."""
    if expr.side_effect_regs():
        return False
    if isinstance(expr, (Call, XRAMRef, IRAMRef, CROMRef)):
        return False
    return all(_is_pure(c) for c in expr.children())


class DeadAssignEliminator(OptimizationPass):
    """
    Remove Assign(Reg(r), pure_rhs) nodes where r is provably dead at that point.
    Uses a single backward pass per block seeded from block.live_out.
    """

    def run(self, func: Function) -> None:
        for block in func.blocks:
            if getattr(block, "_absorbed", False):
                continue
            new_hir = _eliminate_block(block.hir, block.live_out)
            if new_hir is not block.hir:
                block.hir = new_hir


def _eliminate_block(hir, live_out):
    """
    Single backward pass: return a new list with dead pure assignments removed,
    or the original list object if nothing was removed.
    """
    nodes   = list(hir)
    live    = set(live_out)
    to_remove: set = set()

    for i in range(len(nodes) - 1, -1, -1):
        node = nodes[i]
        if (isinstance(node, Assign)
                and isinstance(node.lhs, Regs)
                and node.lhs.is_single
                and node.lhs.name not in live
                and _is_pure(node.rhs)):
            to_remove.add(i)
            # Skip live-set update: removed node contributes no reads or writes.
            continue

        # Keep this node — update the live set (backward step).
        live -= node.written_regs
        live |= node.name_refs()

    if not to_remove:
        return hir
    return [n for i, n in enumerate(nodes) if i not in to_remove]
