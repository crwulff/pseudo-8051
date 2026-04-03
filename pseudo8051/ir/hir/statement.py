"""
ir/hir/statement.py — Statement node (legacy string-based).
"""

from typing import List, Tuple

from pseudo8051.ir.hir._base import HIRNode


class Statement(HIRNode):
    """A single C-like statement string (already formatted by a handler)."""

    def __init__(self, ea: int, text: str):
        super().__init__(ea)
        self.text = text

    @property
    def written_regs(self) -> frozenset:
        eq = self.text.find(" = ")
        if eq > 0:
            lhs_tok = self.text[:eq].split()[-1]
            return frozenset({lhs_tok})
        return frozenset()

    def render(self, indent: int = 0) -> List[Tuple[int, str]]:
        return [(self.ea, f"{self._ind(indent)}{self.text}")]

    def ann_lines(self) -> List[str]:
        return ["Statement", f"  text: {self.text!r}"]
