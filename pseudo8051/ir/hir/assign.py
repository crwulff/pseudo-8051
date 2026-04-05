"""
ir/hir/assign.py — Assign and TypedAssign nodes.
"""

from typing import List, Tuple

from pseudo8051.ir.hir._base import HIRNode, _render_expr, _ann_field, _lhs_written_regs, _refs_from_expr
from pseudo8051.ir.expr import Expr, Regs as RegsExpr


class Assign(HIRNode):
    """lhs = rhs;"""

    def __init__(self, ea: int, lhs: Expr, rhs: Expr):
        super().__init__(ea)
        self.lhs = lhs
        self.rhs = rhs

    @property
    def written_regs(self) -> frozenset:
        return _lhs_written_regs(self.lhs)

    @property
    def use_regs(self) -> frozenset:
        return _refs_from_expr(self.rhs)

    def render(self, indent: int = 0) -> List[Tuple[int, str]]:
        return [(self.ea, f"{self._ind(indent)}{_render_expr(self.lhs)} = {_render_expr(self.rhs)};")]

    def name_refs(self) -> frozenset:
        refs = _refs_from_expr(self.rhs)
        if not isinstance(self.lhs, RegsExpr):
            refs = refs | _refs_from_expr(self.lhs)
        return refs

    def ann_lines(self) -> List[str]:
        return ["Assign"] + _ann_field("lhs", self.lhs) + _ann_field("rhs", self.rhs)


class TypedAssign(Assign):
    """type lhs = rhs;  — typed variable declaration (e.g. retval nodes)."""

    def __init__(self, ea: int, type_str: str, lhs: Expr, rhs: Expr):
        super().__init__(ea, lhs, rhs)
        self.type_str = type_str

    def render(self, indent: int = 0) -> List[Tuple[int, str]]:
        return [(self.ea, f"{self._ind(indent)}{self.type_str} {_render_expr(self.lhs)} = {_render_expr(self.rhs)};")]

    def ann_lines(self) -> List[str]:
        return ["TypedAssign", f"  type: {self.type_str!r}"] + _ann_field("lhs", self.lhs) + _ann_field("rhs", self.rhs)
