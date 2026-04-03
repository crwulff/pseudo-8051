"""
ir/hir/goto_statement.py — GotoStatement node.
"""

from typing import List, Tuple

from pseudo8051.ir.hir._base import HIRNode


class GotoStatement(HIRNode):
    """goto label;"""

    def __init__(self, ea: int, label: str):
        super().__init__(ea)
        self.label = label

    def render(self, indent: int = 0) -> List[Tuple[int, str]]:
        return [(self.ea, f"{self._ind(indent)}goto {self.label};")]

    def ann_lines(self) -> List[str]:
        return ["GotoStatement", f"  label: {self.label!r}"]
