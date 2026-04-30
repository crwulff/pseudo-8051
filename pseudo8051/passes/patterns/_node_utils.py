"""
passes/patterns/_node_utils.py — HIR-node-level expression traversal helpers.

Functions that apply Expr transforms to entire HIR nodes, count register uses,
substitute registers, and fold expressions within nodes.
"""

from typing import Callable, Dict, List, Optional

from pseudo8051.ir.hir import (HIRNode, Assign, CompoundAssign,
                               ExprStmt, ReturnStmt, IfGoto, IfNode, SwitchNode)
from pseudo8051.ir.expr import Expr, Reg, Regs, Name, Const
from pseudo8051.passes.patterns._types import VarInfo
from pseudo8051.passes.patterns._expr_utils import (
    _walk_expr, _fold_unary_const, _canonicalize_expr,
    _subst_pairs_in_expr, _subst_single_regs_in_expr,
)


def _apply_expr_subst_to_node(node: HIRNode,
                               expr_fn: Callable[[Expr], Expr]) -> HIRNode:
    """Apply expr_fn to every read-position Expr in node.

    LHS of assignments is never transformed.  Returns node unchanged if nothing
    changed.  When a new node is created it records the original as its source.
    """
    new_node = node.map_exprs(expr_fn)
    if new_node is node:
        return node
    node.copy_meta_to(new_node)
    new_node.source_nodes = [node]
    return new_node


def _fold_exprs_in_node(node: HIRNode) -> HIRNode:
    """Apply algebraic constant folding to all expression positions in node.

    Useful after substituting a register value to simplify the result
    (e.g. (arg1 * 3) / 3 → arg1).
    """
    return _apply_expr_subst_to_node(node, lambda e: _canonicalize_expr(e, {}, [], {}))


def _replace_pairs_in_node(node: HIRNode,
                            reg_map: Dict[str, "VarInfo"]) -> HIRNode:
    """Apply pair substitution to an Assign / ExprStmt / ReturnStmt rhs/expr."""
    return _apply_expr_subst_to_node(
        node,
        lambda e: _subst_pairs_in_expr(e, reg_map),
    )


def _replace_single_regs_in_node(node: HIRNode,
                                  reg_map: Dict[str, "VarInfo"]) -> HIRNode:
    """Apply single-reg param substitution to RHS/value/expr of a node."""
    return _apply_expr_subst_to_node(
        node,
        lambda e: _subst_single_regs_in_expr(e, reg_map),
    )


def _count_reg_uses_in_node(r: str, node: HIRNode) -> int:
    """Count read-position occurrences of Reg/Name(r) in node.

    Also counts multi-component Regs nodes that contain r as one of their
    individual register names, so that RegGroup(('R4','R5','R6','R7')) is
    counted as a use of 'R4', 'R5', 'R6', and 'R7'.
    """
    count = [0]

    def _fn(e: Expr) -> Expr:
        if e == Reg(r):
            count[0] += 1
        elif isinstance(e, Name) and e.name == r:
            count[0] += 1
        elif isinstance(e, Regs) and not e.is_single and r in e.names:
            count[0] += 1
        return e

    if isinstance(node, Assign):
        _walk_expr(node.rhs, _fn)
        # Also count in compound LHS (e.g. XRAMRef inner), but NOT plain Name
        # (Name LHS is a write destination, not a read; counting it causes
        # _propagate_register_copies to treat the next var=... write as a "use").
        if not isinstance(node.lhs, (Regs, Name)):
            _walk_expr(node.lhs, _fn)
    elif isinstance(node, CompoundAssign):
        _walk_expr(node.rhs, _fn)
    elif isinstance(node, ExprStmt):
        _walk_expr(node.expr, _fn)
    elif isinstance(node, ReturnStmt) and node.value is not None:
        _walk_expr(node.value, _fn)
    elif isinstance(node, IfGoto):
        _walk_expr(node.cond, _fn)
    elif isinstance(node, IfNode):
        _walk_expr(node.condition, _fn)
    elif isinstance(node, SwitchNode):
        _walk_expr(node.subject, _fn)
    return count[0]


def _subst_reg_in_node(node: HIRNode, r: str,
                        replacement: Expr) -> Optional[HIRNode]:
    """
    Replace Reg/Name(r) → replacement in read positions of node.
    Returns updated node, or None if r does not appear.
    """
    def _fn(e: Expr) -> Expr:
        if e == Reg(r):
            return replacement
        if isinstance(e, Name) and e.name == r:
            return replacement
        return _fold_unary_const(e)

    def _out(new_node: HIRNode) -> HIRNode:
        node.copy_meta_to(new_node)
        new_node.source_nodes = [node]
        return new_node

    if isinstance(node, Assign):
        new_rhs = _walk_expr(node.rhs, _fn)
        new_lhs = node.lhs
        # Substitute in compound LHS (e.g. XRAMRef inner) but NOT plain Name
        # (Name LHS is a write destination, not a read position).
        if not isinstance(node.lhs, (Regs, Name)):
            new_lhs = _walk_expr(node.lhs, _fn)
        if new_rhs is node.rhs and new_lhs is node.lhs:
            return None
        return _out(node._rebuild(new_lhs, new_rhs))

    if isinstance(node, CompoundAssign):
        new_rhs = _walk_expr(node.rhs, _fn)
        if new_rhs is node.rhs:
            return None
        return _out(CompoundAssign(node.ea, node.lhs, node.op, new_rhs))

    if isinstance(node, ExprStmt):
        new_expr = _walk_expr(node.expr, _fn)
        if new_expr is node.expr:
            return None
        return _out(ExprStmt(node.ea, new_expr))

    if isinstance(node, ReturnStmt) and node.value is not None:
        new_val = _walk_expr(node.value, _fn)
        if new_val is node.value:
            return None
        return _out(ReturnStmt(node.ea, new_val))

    if isinstance(node, IfGoto):
        new_cond = _walk_expr(node.cond, _fn)
        if new_cond is node.cond:
            return None
        return _out(IfGoto(node.ea, new_cond, node.label))

    if isinstance(node, IfNode):
        new_cond = _walk_expr(node.condition, _fn)
        if new_cond is node.condition:
            return None
        return node.copy_meta_to(IfNode(node.ea, new_cond, node.then_nodes, node.else_nodes))

    if isinstance(node, SwitchNode):
        new_subject = _walk_expr(node.subject, _fn)
        if new_subject is node.subject:
            return None
        return node.copy_meta_to(SwitchNode(node.ea, new_subject, node.cases,
                                             node.default_label, node.default_body,
                                             case_comments=list(node.case_comments),
                                             case_enum_names=list(node.case_enum_names) if node.case_enum_names is not None else None))

    return None


def _fold_into_node(node: HIRNode, name_expr: Expr,
                    replacement: Expr,
                    reg_map: Dict[str, "VarInfo"]) -> Optional[HIRNode]:
    """
    Try to substitute name_expr → replacement in the expression position of node.

    For Assign: substitutes in rhs.
    For ReturnStmt/ExprStmt: substitutes in value/expr.
    Returns None if name_expr does not appear in an expression position.
    """
    name_str = name_expr.render() if isinstance(name_expr, Expr) else str(name_expr)

    def _subst_fn(e: Expr) -> Expr:
        if e == name_expr:
            return replacement
        if isinstance(e, (Name, Regs)) and e.render() == name_str:
            return replacement
        return e

    def _finalize(new_node: HIRNode) -> HIRNode:
        return node.copy_meta_to(_replace_pairs_in_node(new_node, reg_map))

    if isinstance(node, Assign):
        new_rhs = _walk_expr(node.rhs, _subst_fn)
        if new_rhs is node.rhs:
            return None  # not found in rhs
        return _finalize(node._rebuild(node.lhs, new_rhs))

    if isinstance(node, ReturnStmt) and node.value is not None:
        new_val = _walk_expr(node.value, _subst_fn)
        if new_val is node.value:
            return None
        return _finalize(ReturnStmt(node.ea, new_val))

    if isinstance(node, ExprStmt):
        new_expr = _walk_expr(node.expr, _subst_fn)
        if new_expr is node.expr:
            return None
        return _finalize(ExprStmt(node.ea, new_expr))

    return None
