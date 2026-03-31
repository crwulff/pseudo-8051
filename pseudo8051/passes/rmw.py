"""
passes/rmw.py — RMWCollapser: fold XRAM read-modify-write sequences.

Works within each BasicBlock's HIR list.

Recognised pattern (consecutive nodes):

    A = XRAM[addr];          ← Assign(Reg("A"), XRAMRef(inner))
    A &= expr;               ← CompoundAssign(Reg("A"), "&=", rhs)
    A |= expr;               ← CompoundAssign(Reg("A"), "|=", rhs)
    A ^= expr;               ← CompoundAssign(Reg("A"), "^=", rhs)
    A += expr;               ← CompoundAssign(Reg("A"), "+=", rhs)
    A = ~A;                  ← Assign(Reg("A"), UnaryOp("~", Reg("A")))
    XRAM[addr] = A;          ← Assign(XRAMRef(inner), Reg("A"))

One or more modifier nodes between the read and write are required.

Collapsed to:
    XRAM[addr] = <expr_tree>;
"""

from typing import List, Optional, Tuple

from pseudo8051.ir.hir    import HIRNode, Assign, CompoundAssign
from pseudo8051.ir.expr   import Expr, Reg, XRAMRef, UnaryOp, BinOp
from pseudo8051.passes      import OptimizationPass
from pseudo8051.constants   import dbg
from pseudo8051.ir.function import Function

_A = Reg("A")

_OP_MAP = {"&=": "&", "|=": "|", "^=": "^", "+=": "+"}


def _match_read_a(node: HIRNode) -> Optional[Expr]:
    """A = XRAM[inner]; → return inner, else None."""
    if isinstance(node, Assign) and node.lhs == _A and isinstance(node.rhs, XRAMRef):
        return node.rhs.inner
    return None


def _match_op_a(node: HIRNode) -> Optional[Tuple[str, Expr]]:
    """A op= rhs; → return (op_char, rhs_expr), else None."""
    if isinstance(node, CompoundAssign) and node.lhs == _A and node.op in _OP_MAP:
        return (_OP_MAP[node.op], node.rhs)
    return None


def _match_cpl_a(node: HIRNode) -> bool:
    """A = ~A; → True."""
    return (isinstance(node, Assign) and node.lhs == _A
            and isinstance(node.rhs, UnaryOp) and node.rhs.op == "~"
            and node.rhs.operand == _A)


def _match_write_a(node: HIRNode, inner: Expr) -> bool:
    """XRAM[inner] = A; → True (inner must match read's inner)."""
    return (isinstance(node, Assign) and isinstance(node.lhs, XRAMRef)
            and node.lhs.inner == inner and node.rhs == _A)


def _build_expr_node(base: Expr, ops: list) -> Expr:
    """Build the RHS expression tree by applying ops to base."""
    expr = base
    for op_info in ops:
        if op_info[0] == "~":
            expr = UnaryOp("~", expr)
        else:
            op_char, rhs = op_info
            expr = BinOp(expr, op_char, rhs)
    return expr


def _collapse_block_hir(hir: List[HIRNode]) -> List[HIRNode]:
    """Collapse RMW patterns within a single block's HIR list."""
    result: List[HIRNode] = []
    i = 0
    n = len(hir)

    while i < n:
        node = hir[i]

        inner = _match_read_a(node)
        if inner is None:
            result.append(node)
            i += 1
            continue

        # Scan modifier nodes (op or complement)
        ops: list = []
        j = i + 1
        while j < n:
            op_info = _match_op_a(hir[j])
            if op_info is not None:
                ops.append(op_info)
                j += 1
            elif _match_cpl_a(hir[j]):
                ops.append(("~",))
                j += 1
            else:
                break

        if ops and j < n and _match_write_a(hir[j], inner):
            rhs = _build_expr_node(XRAMRef(inner), ops)
            collapsed = Assign(node.ea, XRAMRef(inner), rhs)
            dbg("RMW", f"  {node.render(0)[0][1]!r}  →  {collapsed.render(0)[0][1]!r}")
            result.append(collapsed)
            i = j + 1
            continue

        result.append(node)
        i += 1

    return result


class RMWCollapser(OptimizationPass):
    """Collapse XRAM read-modify-write sequences within each block's HIR."""

    def run(self, func: Function) -> None:
        for block in func.blocks:
            if block.hir:
                before = len(block.hir)
                block.hir = _collapse_block_hir(block.hir)
                after = len(block.hir)
                if after < before:
                    dbg("RMW", f"block {hex(block.start_ea)}: "
                               f"{before - after} line(s) collapsed")
