"""
ir/hir/return_stmt.py — ReturnStmt node.
"""

from typing import List, Optional, Tuple

from pseudo8051.ir.hir._base import HIRNode, _render_expr, _ann_field, _refs_from_expr
from pseudo8051.ir.expr import Expr


class ReturnStmt(HIRNode):
    """return;  or  return expr;  with optional trailing comment."""

    def __init__(self, ea: int, value: Optional[Expr] = None, comment: str = ""):
        super().__init__(ea)
        self.value   = value
        self.comment = comment

    def render(self, indent: int = 0) -> List[Tuple[int, str]]:
        suffix = f"  /* {self.comment} */" if self.comment else ""
        if self.value is None:
            return [(self.ea, f"{self._ind(indent)}return;{suffix}")]
        return [(self.ea, f"{self._ind(indent)}return {_render_expr(self.value)};{suffix}")]

    def name_refs(self) -> frozenset:
        return _refs_from_expr(self.value) if self.value is not None else frozenset()

    def ann_lines(self) -> List[str]:
        return ["ReturnStmt"] + _ann_field("value", self.value)
