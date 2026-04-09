"""
ir/hir/continue_stmt.py — ContinueStmt node.
"""

from typing import List, Tuple

from pseudo8051.ir.hir._base import HIRNode


class ContinueStmt(HIRNode):
    """continue;"""

    def render(self, indent: int = 0) -> List[Tuple[int, str]]:
        return [(self.ea, f"{self._ind(indent)}continue;")]
