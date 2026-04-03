"""
ir/hir/if_goto.py — IfGoto node.
"""

from typing import List, Tuple

from pseudo8051.ir.hir._base import HIRNode, _render_expr, _ann_field
from pseudo8051.ir.expr import Expr


class IfGoto(HIRNode):
    """if (cond) goto label;"""

    def __init__(self, ea: int, cond: Expr, label: str):
        super().__init__(ea)
        self.cond  = cond
        self.label = label

    def render(self, indent: int = 0) -> List[Tuple[int, str]]:
        return [(self.ea, f"{self._ind(indent)}if ({_render_expr(self.cond)}) goto {self.label};")]

    def ann_lines(self) -> List[str]:
        return ["IfGoto"] + _ann_field("cond", self.cond) + [f"  label: {self.label!r}"]
