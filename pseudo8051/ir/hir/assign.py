"""
ir/hir/assign.py — Assign and TypedAssign nodes.
"""

from typing import Callable, List, Tuple

from pseudo8051.ir.hir._base import HIRNode, _render_expr, _render_call_with_comments, _ann_field, _lhs_written_regs, _refs_from_expr
from pseudo8051.ir.expr import Expr, Regs as RegsExpr, Call as CallExpr


class Assign(HIRNode):
    """lhs = rhs;"""

    def __init__(self, ea: int, lhs: Expr, rhs: Expr):
        super().__init__(ea)
        self.lhs = lhs
        self.rhs = rhs

    @property
    def written_regs(self) -> frozenset:
        return _lhs_written_regs(self.lhs)

    def possibly_killed(self) -> frozenset:
        return (self.written_regs
                | self.lhs.side_effect_regs()
                | self.rhs.side_effect_regs())

    @property
    def use_regs(self) -> frozenset:
        return _refs_from_expr(self.rhs)

    def render(self, indent: int = 0) -> List[Tuple[int, str]]:
        rhs_str = (_render_call_with_comments(self.rhs, self.ann)
                   if isinstance(self.rhs, CallExpr)
                   else _render_expr(self.rhs))
        return [(self.ea, f"{self._ind(indent)}{_render_expr(self.lhs)} = {rhs_str};")]

    def name_refs(self) -> frozenset:
        refs = _refs_from_expr(self.rhs)
        if not isinstance(self.lhs, RegsExpr):
            refs = refs | _refs_from_expr(self.lhs)
        return refs

    def map_exprs(self, fn: Callable[[Expr], Expr]) -> "Assign":
        new_rhs = fn(self.rhs)
        if new_rhs is self.rhs:
            return self
        return Assign(self.ea, self.lhs, new_rhs)

    def _rebuild(self, new_lhs: Expr, new_rhs: Expr) -> "Assign":
        """Return a copy of this node with new_lhs/new_rhs, preserving subclass metadata."""
        return Assign(self.ea, new_lhs, new_rhs)

    def ann_lines(self) -> List[str]:
        return ["Assign"] + _ann_field("lhs", self.lhs) + _ann_field("rhs", self.rhs)


class TypedAssign(Assign):
    """type lhs = rhs;  — typed variable declaration (e.g. retval nodes)."""

    def __init__(self, ea: int, type_str: str, lhs: Expr, rhs: Expr):
        super().__init__(ea, lhs, rhs)
        self.type_str = type_str

    def render(self, indent: int = 0) -> List[Tuple[int, str]]:
        rhs_str = (_render_call_with_comments(self.rhs, self.ann)
                   if isinstance(self.rhs, CallExpr)
                   else _render_expr(self.rhs))
        return [(self.ea, f"{self._ind(indent)}{self.type_str} {_render_expr(self.lhs)} = {rhs_str};")]

    def map_exprs(self, fn: Callable[[Expr], Expr]) -> "TypedAssign":
        new_rhs = fn(self.rhs)
        if new_rhs is self.rhs:
            return self
        return TypedAssign(self.ea, self.type_str, self.lhs, new_rhs)

    def _rebuild(self, new_lhs: Expr, new_rhs: Expr) -> "TypedAssign":
        return TypedAssign(self.ea, self.type_str, new_lhs, new_rhs)

    def ann_lines(self) -> List[str]:
        return ["TypedAssign", f"  type: {self.type_str!r}"] + _ann_field("lhs", self.lhs) + _ann_field("rhs", self.rhs)
