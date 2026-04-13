"""
ir/hir/expr_stmt.py — ExprStmt node.
"""

from typing import List, Tuple

from pseudo8051.ir.hir._base import HIRNode, _render_expr, _ann_field, _refs_from_expr
from pseudo8051.ir.expr import Expr, UnaryOp, Regs


class ExprStmt(HIRNode):
    """A standalone expression statement: push(R7);  R7++;"""

    def __init__(self, ea: int, expr: Expr):
        super().__init__(ea)
        self.expr = expr

    def render(self, indent: int = 0) -> List[Tuple[int, str]]:
        return [(self.ea, f"{self._ind(indent)}{_render_expr(self.expr)};")]

    @property
    def written_regs(self) -> frozenset:
        return frozenset()

    def possibly_killed(self) -> frozenset:
        """For ExprStmt(reg++/--), the register is modified as a side effect."""
        if (isinstance(self.expr, UnaryOp)
                and self.expr.op in ('++', '--')
                and isinstance(self.expr.operand, Regs)
                and self.expr.operand.is_single):
            r = self.expr.operand.name
            if r == 'DPTR':
                return frozenset({'DPTR', 'DPH', 'DPL'})
            return frozenset({r})
        return frozenset()

    def name_refs(self) -> frozenset:
        return _refs_from_expr(self.expr)

    def ann_lines(self) -> List[str]:
        return ["ExprStmt"] + _ann_field("expr", self.expr)
