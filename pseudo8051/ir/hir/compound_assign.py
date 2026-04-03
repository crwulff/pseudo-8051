"""
ir/hir/compound_assign.py — CompoundAssign node.
"""

from typing import List, Tuple

from pseudo8051.ir.hir._base import HIRNode, _render_expr, _ann_field, _lhs_written_regs, _refs_from_expr
from pseudo8051.ir.expr import Expr, Reg as RegExpr, Name as NameExpr


class CompoundAssign(HIRNode):
    """lhs op= rhs;  e.g. A += rhs;"""

    def __init__(self, ea: int, lhs: Expr, op: str, rhs: Expr):
        super().__init__(ea)
        self.lhs = lhs
        self.op  = op
        self.rhs = rhs

    @property
    def written_regs(self) -> frozenset:
        return _lhs_written_regs(self.lhs)

    def render(self, indent: int = 0) -> List[Tuple[int, str]]:
        return [(self.ea, f"{self._ind(indent)}{_render_expr(self.lhs)} {self.op} {_render_expr(self.rhs)};")]

    def name_refs(self) -> frozenset:
        refs = _refs_from_expr(self.rhs)
        # CompoundAssign also READS its LHS operand
        if isinstance(self.lhs, (RegExpr, NameExpr)):
            refs = refs | frozenset({self.lhs.name})
        return refs

    def ann_lines(self) -> List[str]:
        return (["CompoundAssign"] + _ann_field("lhs", self.lhs)
                + [f"  op: {self.op!r}"] + _ann_field("rhs", self.rhs))
