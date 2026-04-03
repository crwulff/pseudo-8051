"""
ir/hir/break_stmt.py — BreakStmt node.
"""

from typing import List, Tuple

from pseudo8051.ir.hir._base import HIRNode


class BreakStmt(HIRNode):
    """break;"""

    def render(self, indent: int = 0) -> List[Tuple[int, str]]:
        return [(self.ea, f"{self._ind(indent)}break;")]
