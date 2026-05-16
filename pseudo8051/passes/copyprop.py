"""
passes/copyprop.py — CopyPropagator: forward intra-block copy propagation.

For each simple assignment of the form:
    Assign(Reg(r), simple_expr)   where simple_expr is a Reg or Const

records that r == simple_expr and substitutes Reg(r) → simple_expr in all
subsequent read positions in the same block, until:
  • r is overwritten (new assignment to r)
  • a source register is overwritten (invalidates copies that used it)
  • a function call is encountered (kills all copies)

Only Reg→Reg and Reg→Const copies are tracked to avoid incorrect substitutions
across potential memory writes (Reg→XRAMRef, etc. are intentionally excluded).

Runs before AnnotationPass and RMWCollapser so structural passes see clean HIR.
"""

from typing import Dict, List

from pseudo8051.ir.function import Function
from pseudo8051.ir.hir     import HIRNode, Assign
from pseudo8051.ir.hir.expr_stmt      import ExprStmt
from pseudo8051.ir.hir.if_goto        import IfGoto
from pseudo8051.ir.hir.compound_assign import CompoundAssign
from pseudo8051.ir.expr    import Expr, Regs, Const, Call, UnaryOp
from pseudo8051.passes     import OptimizationPass


def _is_copy_worthy(expr: Expr) -> bool:
    """True if expr is a Reg or Const — safe to forward-propagate unconditionally."""
    return isinstance(expr, (Regs, Const))


def _expr_has_call(expr: Expr) -> bool:
    """True if expr is or contains a function Call."""
    if isinstance(expr, Call):
        return True
    return any(_expr_has_call(c) for c in expr.children())


def _node_has_call(node: HIRNode) -> bool:
    """True if any top-level expression in node is or contains a Call."""
    if isinstance(node, (Assign, CompoundAssign)):
        return _expr_has_call(node.rhs)
    if isinstance(node, ExprStmt):
        return _expr_has_call(node.expr)
    if isinstance(node, IfGoto):
        return _expr_has_call(node.cond)
    return False


def _invalidate_source(copies: Dict[str, Expr], reg: str) -> None:
    """Remove copies whose value is Reg(reg) — source was overwritten."""
    to_del = [k for k, v in copies.items()
              if isinstance(v, Regs) and v.is_single and v.name == reg]
    for k in to_del:
        del copies[k]


class CopyPropagator(OptimizationPass):
    """
    Forward copy propagation within each BasicBlock's flat HIR list.
    Operates before structural passes (all HIR nodes are leaf-level).
    """

    def run(self, func: Function) -> None:
        for block in func.blocks:
            if getattr(block, "_absorbed", False):
                continue
            _propagate_block(block)


def _propagate_block(block) -> None:
    copies: Dict[str, Expr] = {}
    new_hir: List[HIRNode] = []
    changed = False

    def _subst(expr: Expr) -> Expr:
        """Substitute single-register reads using the current copies dict."""
        # Don't substitute inside ++/--.  The operand is both read and written;
        # replacing Reg("DPTR") with Const(0xdc68) would turn "DPTR++" into
        # "EXT_DC68++" which no longer increments the DPTR register.
        if isinstance(expr, UnaryOp) and expr.op in ('++', '--'):
            return expr
        if isinstance(expr, Regs) and expr.is_single:
            rep = copies.get(expr.name)
            if rep is not None:
                return rep
        children = expr.children()
        if not children:
            return expr
        new_children = [_subst(c) for c in children]
        if all(n is o for n, o in zip(new_children, children)):
            return expr
        return expr.rebuild(new_children)

    for node in block.hir:
        # Step 1: substitute copies into all read positions of this node
        new_node = node.map_exprs(_subst)
        if new_node is not node:
            changed = True
        new_hir.append(new_node)
        node = new_node  # use new node for copy tracking below

        # Step 2: any function call kills all copies (calls clobber registers)
        if _node_has_call(node):
            copies.clear()

        # Step 3: update copies dict for register writes
        if (isinstance(node, Assign)
                and isinstance(node.lhs, Regs)
                and node.lhs.is_single
                and _is_copy_worthy(node.rhs)):
            r = node.lhs.name
            # Invalidate copies whose source was r (r is being redefined)
            _invalidate_source(copies, r)
            copies[r] = node.rhs
        else:
            # Invalidate copies for all registers written or side-effected by this
            # node.  possibly_killed() covers both LHS writes (written_regs) and
            # side-effect writes from ++/-- on registers (e.g. DPTR++ kills DPTR,
            # DPH, DPL even though ExprStmt.written_regs is empty).
            for w in node.possibly_killed():
                _invalidate_source(copies, w)
                copies.pop(w, None)

    if changed:
        block.hir = new_hir
