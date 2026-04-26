"""
ir/hir/expr_stmt.py — ExprStmt node.
"""

from typing import List, Tuple

from pseudo8051.ir.hir._base import HIRNode, _render_expr, _render_call_with_comments, _ann_field, _refs_from_expr
from pseudo8051.ir.expr import Expr, Call as CallExpr


class ExprStmt(HIRNode):
    """A standalone expression statement: push(R7);  R7++;"""

    def __init__(self, ea: int, expr: Expr):
        super().__init__(ea)
        self.expr = expr

    def render(self, indent: int = 0) -> List[Tuple[int, str]]:
        if isinstance(self.expr, CallExpr):
            text = _render_call_with_comments(self.expr, self.ann)
        else:
            text = _render_expr(self.expr)
        return [(self.ea, f"{self._ind(indent)}{text};")]

    @property
    def written_regs(self) -> frozenset:
        return frozenset()

    def possibly_killed(self) -> frozenset:
        """Delegate to the expression's side-effect register set."""
        return self.expr.side_effect_regs()

    def name_refs(self) -> frozenset:
        return _refs_from_expr(self.expr)

    def ann_lines(self) -> List[str]:
        return ["ExprStmt"] + _ann_field("expr", self.expr)
