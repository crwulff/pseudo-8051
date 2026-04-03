"""
ir/hir/label.py — Label node.
"""

from typing import List, Tuple

from pseudo8051.ir.hir._base import HIRNode


class Label(HIRNode):
    """label_XXXX: — emitted before a block that needs a label."""

    def __init__(self, ea: int, name: str):
        super().__init__(ea)
        self.name = name

    def render(self, indent: int = 0) -> List[Tuple[int, str]]:
        return [
            (self.ea, ""),
            (self.ea, f"{self.name}:"),
        ]

    def ann_lines(self) -> List[str]:
        return ["Label", f"  name: {self.name!r}"]
