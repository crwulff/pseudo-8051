"""
passes/typesimplify/_propagate_utils.py — Shared helpers for the propagation sub-passes.

Small utility functions used by _propagate_regcopy and _propagate_inline; not
intended for direct use by callers outside the typesimplify package.
"""

from typing import Dict, List, Optional, Tuple

from pseudo8051.ir.hir import (HIRNode, Assign, TypedAssign, CompoundAssign,
                                ExprStmt, ReturnStmt, IfGoto, IfNode,
                                SwitchNode)
from pseudo8051.ir.expr import (Expr, Call, Const, XRAMRef, UnaryOp,
                                 Regs as RegExpr, Name as NameExpr)
from pseudo8051.passes.patterns._utils import VarInfo, _count_reg_uses_in_node, _walk_expr
from pseudo8051.constants import dbg


# ── Compound-assign op map ────────────────────────────────────────────────────

_COMPOUND_OPS = {"+=": "+", "-=": "-", "&=": "&", "|=": "|", "^=": "^"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _has_xram_const_addr(node: HIRNode, val: int) -> bool:
    """True if node or a direct source_node has XRAMRef(Const(val)) as lhs."""
    def _check(n: HIRNode) -> bool:
        return (isinstance(n, Assign)
                and isinstance(n.lhs, XRAMRef)
                and isinstance(n.lhs.inner, Const)
                and n.lhs.inner.value == val)
    if _check(node):
        return True
    for sn in getattr(node, 'source_nodes', []):
        if _check(sn):
            return True
    return False


def _name_possibly_written_in(name: str, node: HIRNode) -> bool:
    """Return True if any node inside node's bodies may assign to Name(name)."""
    def _check_seq(nodes) -> bool:
        for n in nodes:
            lhs = getattr(n, 'lhs', None)
            if isinstance(lhs, NameExpr) and lhs.name == name:
                return True
            for _extra, body in n.child_body_groups():
                if _check_seq(body):
                    return True
            if isinstance(n, SwitchNode):
                for _vals, body in n.cases:
                    if isinstance(body, list) and _check_seq(body):
                        return True
                if isinstance(n.default_body, list) and _check_seq(n.default_body):
                    return True
        return False

    for _extra, body in node.child_body_groups():
        if _check_seq(body):
            return True
    if isinstance(node, SwitchNode):
        for _vals, body in node.cases:
            if isinstance(body, list) and _check_seq(body):
                return True
        if isinstance(node.default_body, list) and _check_seq(node.default_body):
            return True
    return False


def _expr_name_refs(expr: Expr) -> frozenset:
    """Collect register/name identities referenced in an expression tree."""
    refs: set = set()

    def _collect(e: Expr) -> Expr:
        if isinstance(e, RegExpr):
            refs.update(e.regs)
        elif isinstance(e, NameExpr):
            refs.add(e.name)
        return e

    _walk_expr(expr, _collect)
    return frozenset(refs)


def _count_group_uses_in_node(regs_tuple: tuple, node: HIRNode) -> int:
    """Count occurrences of Regs(regs_tuple) used as a full-group expression in node."""
    count = [0]

    def _fn(e: Expr) -> Expr:
        if isinstance(e, RegExpr) and not e.is_single and e.names == regs_tuple:
            count[0] += 1
        return e

    if isinstance(node, Assign):
        _walk_expr(node.rhs, _fn)
        if not isinstance(node.lhs, (RegExpr, NameExpr)):
            _walk_expr(node.lhs, _fn)
    elif isinstance(node, (CompoundAssign, ExprStmt)):
        _walk_expr(getattr(node, 'rhs', None) or node.expr, _fn)
    elif isinstance(node, ReturnStmt) and node.value is not None:
        _walk_expr(node.value, _fn)
    elif isinstance(node, (IfGoto, IfNode)):
        _walk_expr(getattr(node, 'cond', None) or node.condition, _fn)
    return count[0]


def _xram_pre_incr_delta(node: HIRNode, r: str) -> Optional[int]:
    """If node is Assign(XRAMRef(++/-- r or r++/r--), ...), return +1 or -1."""
    if not isinstance(node, Assign):
        return None
    lhs = node.lhs
    if not isinstance(lhs, XRAMRef):
        return None
    inner = lhs.inner
    if not (isinstance(inner, UnaryOp)
            and isinstance(inner.operand, RegExpr)
            and inner.operand.is_single
            and inner.operand.name == r):
        return None
    return +1 if inner.op == '++' else -1


def _expr_stmt_incr_delta(node: HIRNode, r: str) -> Optional[int]:
    """If node is ExprStmt(r++ / r-- / ++r / --r), return +1 or -1."""
    if not isinstance(node, ExprStmt):
        return None
    expr = node.expr
    if not (isinstance(expr, UnaryOp)
            and isinstance(expr.operand, RegExpr)
            and expr.operand.is_single
            and expr.operand.name == r):
        return None
    return +1 if expr.op == '++' else -1


def _collect_mid_writes(nodes: List[HIRNode], reg_map: Dict) -> frozenset:
    """
    Collect all names/regs written by a sequence of nodes, with register↔variable
    cross-expansion via reg_map.
    """
    result: set = set()
    for node in nodes:
        node_writes = node.possibly_killed()
        result.update(node_writes)
        if reg_map:
            for reg in node_writes:
                info = reg_map.get(reg)
                if isinstance(info, VarInfo) and info.name:
                    result.add(info.name)
        if isinstance(node, (Assign, TypedAssign)):
            lhs = getattr(node, 'lhs', None)
            if isinstance(lhs, NameExpr):
                result.add(lhs.name)
                if reg_map:
                    for reg_key, info in reg_map.items():
                        if isinstance(info, VarInfo) and info.name == lhs.name:
                            result.add(reg_key)
    if 'DPH' in result or 'DPL' in result:
        result.add('DPTR')
    if 'DPTR' in result:
        result.add('DPH')
        result.add('DPL')
    return frozenset(result)


def _as_retval_stmt(node: HIRNode) -> Optional[Tuple[str, Call]]:
    """Return (retval_name, call_expr) if node is a TypedAssign retval node; else None."""
    if isinstance(node, TypedAssign) and isinstance(node.rhs, Call):
        return (node.lhs.render(), node.rhs)
    return None


def _count_name_uses_in_nodes(name: str, nodes: List[HIRNode]) -> int:
    """Count total occurrences of Name/Reg(name) in read positions across nodes."""
    return sum(_count_reg_uses_in_node(name, n) for n in nodes)


def _collect_hir_name_refs(nodes: List[HIRNode]) -> frozenset:
    result: set = set()
    for n in nodes:
        result |= n.name_refs()
    return frozenset(result)


def _dbg_node(n) -> str:
    try:
        return f"{type(n).__name__}:{n.render(0)[0][1][:50]!r}"
    except Exception:
        return type(n).__name__
